#!/usr/bin/env python3
"""Check: ml4_pack.py's rank packs read back and every layout undone, compared byte for byte with the source
pack: dtr entries -> trellis slices, F16W records -> fp16 rows, UK / UV -> kv_b rows (and the zero padding), norms,
ssmax, indexer tensors, the head shards (fp16 W_eff against a fresh dequant of the rank's columns, rel ~1e-4 is fp16).
The routed experts (tr3.*) were checked by P3b's kernel test; here only their table and sizes are checked.
Usage: ml4_pack_check.py PACK_DIR OUT_DIR NR [EXL3_REF_DIR]   (CHECK_RANKS=2: only those ranks' packs)"""
import json
import os
import struct
import sys

import numpy as np

pack, out, NR = sys.argv[1], sys.argv[2], int(sys.argv[3])
index = json.load(open(os.path.join(pack, 'model.safetensors.index.json')))['weight_map']
tc = json.load(open(os.path.join(pack, 'config.json')))
tc = tc.get('text_config', tc)
DT = {'I16': np.int16, 'F16': np.float16, 'I32': np.int32, 'BF16': np.uint16}
ND = {'u8': np.uint8, 'f16': np.float16, 'bf16': np.uint16, 'f32': np.float32, 'i64': np.int64, 'i16': np.int16, 'i32': np.int32}


def load(name):
    with open(os.path.join(pack, index[name]), 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        m = json.loads(f.read(n))[name]
        s, e = m['data_offsets']
        f.seek(8 + n + s)
        return np.frombuffer(f.read(e - s), DT[m['dtype']]).reshape(m['shape'])


GV = []
for l_ in range(32):
    r_, c_ = l_ // 4, l_ % 4
    GV += [(r_, 2 * c_), (r_, 2 * c_ + 1), (r_ + 8, 2 * c_), (r_ + 8, 2 * c_ + 1), (r_, 2 * c_ + 8), (r_, 2 * c_ + 9), (r_ + 8, 2 * c_ + 8), (r_ + 8, 2 * c_ + 9)]
GV = np.array(GV)


def unrecord(rec, O, K):
    t = rec.view(np.float16).reshape(O // 16, K // 16, 256)
    w = np.zeros((O // 16, K // 16, 16, 16), np.float16)
    w[:, :, GV[:, 0], GV[:, 1]] = t
    return w.transpose(0, 2, 1, 3).reshape(O, K)


def chunk(n, l, r):
    w = [n // NR + (1 if j < n % NR else 0) for j in range(NR)]
    j = (r - l) % NR
    return sum(w[:j]), w[j]


def bf(x):
    return x if x.dtype == np.uint16 else (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


bad = []


def eq(label, a, b):
    if not (a.shape == b.shape and np.array_equal(a, b)):
        bad.append(label)
        print('  MISMATCH', label, a.shape, b.shape, flush=True)


for r in ([int(x) for x in os.environ['CHECK_RANKS'].split(',')] if os.environ.get('CHECK_RANKS') else range(NR)):
    man = json.load(open(os.path.join(out, f'rank{r}.manifest.json')))
    T = {t['name']: t for t in man['tensors']}
    f = open(os.path.join(out, f'rank{r}.pack'), 'rb')

    def get(name):
        t = T[name]
        f.seek(t['offset'])
        return np.frombuffer(f.read(t['nbytes']), ND[t['dtype']]).reshape(t['shape'])
    for l in man['layers']:
        P, S = f'L{l:03d}', f'layers.{l}.'
        rope, own, full = tc['no_rope_layers'][l] == 1, tc['kv_source_layers'][l] == l, tc['indexer_types'][l] == 'full'
        nope = 64 if rope else 128
        h0, nh = chunk(64, l, r)
        eq(f'r{r} {P} attn_norm', get(P + '.attn_norm'), load(S + 'attention_norm.weight'))
        eq(f'r{r} {P} mlp_norm', get(P + '.mlp_norm'), load(S + 'ffn_norm.weight'))
        if own:
            kva = load(S + 'attention.wkv_a_with_mqa.weight')
            w = unrecord(get(P + '.kv_a.f16.gv'), 576, 6144)
            eq(f'r{r} {P} kv_a', w[:kva.shape[0] if kva.shape[0] < 576 else 576], kva[:576])
            if kva.shape[0] < 576:
                eq(f'r{r} {P} kv_a pad', w[512:], np.zeros((64, 6144), np.float16))
            eq(f'r{r} {P} kv_a_norm', get(P + '.kv_a_norm'), load(S + 'attention.kv_a_norm.weight'))
        else:
            eq(f'r{r} {P} no kv_a', np.array([P + '.kv_a.f16.gv' in T]), np.array([False]))
        kvb = load(S + 'attention.wkv_b.weight').reshape(64, nope + 128, 512)
        uk, uv = get(P + '.kv_b.uk'), get(P + '.kv_b.uv')
        eq(f'r{r} {P} uk', uk[:, :nope], bf(kvb[h0:h0 + nh, :nope]))
        eq(f'r{r} {P} uk pad', uk[:, nope:], np.zeros_like(uk[:, nope:]))
        eq(f'r{r} {P} uv', uv[:, :128], bf(kvb[h0:h0 + nh, nope:]))
        eq(f'r{r} {P} uv pad', uv[:, 128:], np.zeros_like(uv[:, 128:]))
        for nm, pre, side, (c0, c1) in [('q_b', 'attention.wq', 1, (128 * h0, 128 * (h0 + nh))),
                                        ('o_proj', 'attention.wo', 2, (128 * h0, 128 * (h0 + nh)))] + (
                [(f'dense.{d}', f'feed_forward.{w_}', s, tuple(128 * x for x in (lambda c: (c[0], c[0] + c[1]))(chunk(192, l, r))))
                 for d, w_, s in (('gate', 'w1', 1), ('up', 'w3', 1), ('down', 'w2', 2))] if l == 0 else []):
            tr, suh, svh = load(S + pre + '.trellis'), load(S + pre + '.suh'), load(S + pre + '.svh')
            KT, NT, W = tr.shape
            ent = get(f'{P}.{nm}.trellis').view(np.int16)
            if side == 1:
                back = ent.reshape(-1, KT, 8, W).transpose(1, 0, 2, 3).reshape(KT, -1, W)
                eq(f'r{r} {P} {nm} trellis', back, tr[:, c0 // 16:c1 // 16])
                eq(f'r{r} {P} {nm} svh', get(f'{P}.{nm}.svh'), svh[c0:c1])
                eq(f'r{r} {P} {nm} suh', get(f'{P}.{nm}.suh'), suh)
            else:
                back = ent.reshape(-1, NT // 8, 8, 8, W).transpose(0, 2, 1, 3, 4).reshape(-1, NT, W)
                eq(f'r{r} {P} {nm} trellis', back, tr[c0 // 16:c1 // 16])
                eq(f'r{r} {P} {nm} suh', get(f'{P}.{nm}.suh'), suh[c0:c1])
                eq(f'r{r} {P} {nm} svh', get(f'{P}.{nm}.svh'), svh)
        if full:
            I = S + 'attention.indexer.'
            eq(f'r{r} {P} idx wq_b', unrecord(get(P + '.idx.wq_b.f16.gv'), 4096, 6144), load(I + 'wq_b.weight'))
            eq(f'r{r} {P} idx wk', unrecord(get(P + '.idx.wk.f16.gv'), 128, 6144), load(I + 'wk.weight'))
            eq(f'r{r} {P} idx wp', unrecord(get(P + '.idx.weights_proj.f16.gv'), 32, 6144), load(I + 'weights_proj.weight'))
            eq(f'r{r} {P} idx k_norm', np.stack([get(P + '.idx.k_norm.w'), get(P + '.idx.k_norm.b')]),
               np.stack([bf(load(I + 'k_norm.weight')), bf(load(I + 'k_norm.bias'))]))
            eq(f'r{r} {P} idx q_norm', get(P + '.idx.q_norm'), np.stack([load(I + 'q_input_norm.weight'), load(I + 'q_input_norm.bias')]).astype(np.float32))
        if l >= 1:
            eq(f'r{r} {P} router', unrecord(get(P + '.router.f16.gv'), 128, 6144), load(S + 'gate.weight'))
            tab = get(P + '.tr3.table')
            eq(f'r{r} {P} tr3 rows', np.array([tab.shape[0], int(tab[:, 3].sum())]), np.array([129, 129 * 48 // NR]))
    if 'G.lm_head.f16.gv' in T:
        eq(f'r{r} embed', get('G.embed'), load('tok_embeddings.weight'))
        eq(f'r{r} final_norm', get('G.final_norm'), load('norm.weight'))
        if len(sys.argv) > 4:
            sys.path.insert(0, sys.argv[4])
            import torch
            import exl3_ref
            VN = 204800 // NR
            v0 = VN * r
            cols = slice(v0, v0 + 1024)      # first 1024 vocab rows of the shard: fresh fp32 dequant against the stored fp16
            ref = exl3_ref.dequant_exl3(torch.from_numpy(load('output.trellis')[:, v0 // 16:(v0 + 1024) // 16].copy()),
                                        torch.from_numpy(load('output.suh').copy()), torch.from_numpy(load('output.svh')[cols].copy()), mul1=0x83DCD12D)
            mine = unrecord(get('G.lm_head.f16.gv')[:64], 1024, 6144).astype(np.float32)
            rel = float(np.linalg.norm(mine - ref.T.numpy()) / np.linalg.norm(ref.numpy()))
            print(f'r{r} lm_head rows {v0}-{v0 + 1023}: fp16 vs fp32 W_eff rel {rel:.2e}', flush=True)
            if rel > 1e-3:
                bad.append(f'r{r} lm_head')
    print(f'rank {r}: checked', flush=True)
print('pack check:', 'PASS' if not bad else f'FAIL {len(bad)}: {bad[:8]}')
sys.exit(1 if bad else 0)
