#!/usr/bin/env python3
"""ML4's DSpark draft head (the pack's draft/ shard, 38 mtp_dspark.* BF16 tensors) -> one gr8 pack per
rank. Rank r holds:
  DS.tp.gv              BF16W record [1536 / 16][30720 / 16][512]  target_projection rows 1536 r .. (output quarter)
  DS.L<j>.wq.gv         [2048 / 16][6144 / 16]   wq rows of heads 16 r .. 16 r + 15 (128 each)
  DS.L<j>.kva.gv        [576 / 16][6144 / 16]    wkv_a_with_mqa (whole: every rank forms the latent)
  DS.L<j>.kvb.gv        [3072 / 16][512 / 16]    wkv_b rows of those heads (192 each: 64 k_nope + 128 v)
  DS.L<j>.wo.gv         [6144 / 16][2048 / 16]   wo columns of those heads
  DS.L<j>.w1.gv, .w3.gv [6144 / 16][6144 / 16]   FFN rows 6144 r ..;  DS.L<j>.w2.gv [6144 / 16][6144 / 16] its columns
  DS.L<j>.{attn_norm, ffn_norm} bf16 [6144], DS.L<j>.kv_norm bf16 [512]
  DS.jp.gv              [1536 / 16][7168 / 16]   autoregressive_head.joint_proj (whole), DS.jp_bias bf16 [1536]
  DS.ar_out.gv          [51200 / 16][512 / 16]   autoregressive_head.output rows 51200 r .. (this rank's vocab slice)
  DS.ar_emb             bf16 [204800][512]       autoregressive_head.embedding (whole: row gathers)
  DS.mask bf16 [2][6144], DS.target_norm, DS.final_norm bf16 [6144]
BF16W records are ml4_pack.py's f16_record order on the bf16 bits (gr8's BF16W GEMV, bf16 fragments).
Usage: ml4_dspark_pack.py PACK_DIR GR8_SRC NR OUT_DIR [--ranks 0,1]"""
import argparse
import json
import os
import struct
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('pack')
ap.add_argument('gr8')
ap.add_argument('nr', type=int)
ap.add_argument('out')
ap.add_argument('--ranks', default='')
a = ap.parse_args()
sys.path.insert(0, a.gr8)
from tools.convert.pack import PackWriter  # noqa: E402

NR = a.nr
assert NR == 4, 'the DSpark split is written for TP4 (64 heads / 4, vocab 204800 / 4)'
ranks = [int(r) for r in a.ranks.split(',')] if a.ranks else list(range(NR))
DIR = os.path.join(a.pack, 'draft')
src = [os.path.join(DIR, f) for f in sorted(os.listdir(DIR)) if f.endswith('.safetensors')]
where = {}
for p in src:
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for k, v in json.loads(f.read(n)).items():
            if k != '__metadata__':
                where[k] = (p, 8 + n, v)
assert sum(k.startswith('mtp_dspark.') for k in where) == 38, 'expected the 38 mtp_dspark tensors'


def load(name):
    p, base, m = where['mtp_dspark.' + name]
    assert m['dtype'] == 'BF16', (name, m['dtype'])
    s, e = m['data_offsets']
    with open(p, 'rb') as f:
        f.seek(base + s)
        return np.frombuffer(f.read(e - s), np.uint16).reshape(m['shape'])


GV = []
for l_ in range(32):
    r_, c_ = l_ // 4, l_ % 4
    GV += [(r_, 2 * c_), (r_, 2 * c_ + 1), (r_ + 8, 2 * c_), (r_ + 8, 2 * c_ + 1),
           (r_, 2 * c_ + 8), (r_, 2 * c_ + 9), (r_ + 8, 2 * c_ + 8), (r_ + 8, 2 * c_ + 9)]
GV = np.array(GV)


def record(w):
    """bf16 bits [O][K] -> u8 [O/16][K/16][512] (the BF16W / F16W record order)."""
    w = np.ascontiguousarray(w, np.uint16)
    O, K = w.shape
    assert O % 16 == 0 and K % 16 == 0, w.shape
    t = w.reshape(O // 16, 16, K // 16, 16).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t[:, :, GV[:, 0], GV[:, 1]]).view(np.uint8)


HEADS, HD = 64, 128
hr = HEADS // NR
os.makedirs(a.out, exist_ok=True)
for r in ranks:
    w = PackWriter(os.path.join(a.out, f'rank{r}.pack'), r, False, dict(converter='ml4_dspark_pack', model='mistral_large4_dspark', nr=NR))
    put = lambda n, x, dt='u8', **i: w.put(n, x, dt, **i)
    tp = load('backbone.target_projection.weight')                 # [6144][30720]
    q = tp.shape[0] // NR
    put('DS.tp.gv', record(tp[q * r:q * (r + 1)]), layout='BF16W record, target_projection output quarter')
    put('DS.target_norm', load('backbone.target_norm.weight'), 'bf16')
    put('DS.final_norm', load('backbone.final_norm.weight'), 'bf16')
    put('DS.mask', load('backbone.mask_embedding'), 'bf16')
    for j in range(3):
        L = f'backbone.layers.{j}.'
        wq = load(L + 'attention.wq.weight')                       # [8192][6144]
        put(f'DS.L{j}.wq.gv', record(wq[hr * HD * r:hr * HD * (r + 1)]))
        put(f'DS.L{j}.kva.gv', record(load(L + 'attention.wkv_a_with_mqa.weight')))
        kvb = load(L + 'attention.wkv_b.weight')                   # [12288][512]
        put(f'DS.L{j}.kvb.gv', record(kvb[hr * 192 * r:hr * 192 * (r + 1)]))
        wo = load(L + 'attention.wo.weight')                       # [6144][8192]
        put(f'DS.L{j}.wo.gv', record(wo[:, hr * HD * r:hr * HD * (r + 1)]))
        w1, w3, w2 = load(L + 'feed_forward.w1.weight'), load(L + 'feed_forward.w3.weight'), load(L + 'feed_forward.w2.weight')
        f = w1.shape[0] // NR
        put(f'DS.L{j}.w1.gv', record(w1[f * r:f * (r + 1)]))
        put(f'DS.L{j}.w3.gv', record(w3[f * r:f * (r + 1)]))
        put(f'DS.L{j}.w2.gv', record(w2[:, f * r:f * (r + 1)]))
        put(f'DS.L{j}.attn_norm', load(L + 'attention_norm.weight'), 'bf16')
        put(f'DS.L{j}.ffn_norm', load(L + 'ffn_norm.weight'), 'bf16')
        put(f'DS.L{j}.kv_norm', load(L + 'attention.kv_a_norm.weight'), 'bf16')
    put('DS.jp.gv', record(load('autoregressive_head.joint_proj.weight')))
    put('DS.jp_bias', load('autoregressive_head.joint_proj.bias'), 'bf16')
    out = load('autoregressive_head.output.weight')                # [204800][512]
    v = out.shape[0] // NR
    put('DS.ar_out.gv', record(out[v * r:v * (r + 1)]), layout='BF16W record, this rank vocab slice')
    put('DS.ar_emb', load('autoregressive_head.embedding.weight'), 'bf16')
    w.close()
    print(f'rank {r}: {os.path.getsize(os.path.join(a.out, f"rank{r}.pack")) / 1e9:.2f} GB', flush=True)
