#!/usr/bin/env python3
"""Check: ml4_dspark_pack.py's rank packs read back, every BF16W record undone (the inverse of
the f16_record order) and compared byte for byte with the source slice of the draft shard; plain tensors likewise.
Usage: ml4_dspark_check.py PACK_DIR OUT_DIR NR RANK"""
import json
import os
import struct
import sys

import numpy as np

pack, out, NR, r = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
DIR = os.path.join(pack, 'draft')
where = {}
for f in sorted(os.listdir(DIR)):
    if f.endswith('.safetensors'):
        p = os.path.join(DIR, f)
        with open(p, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            for k, v in json.loads(fh.read(n)).items():
                if k != '__metadata__':
                    where[k] = (p, 8 + n, v)


def src(name):
    p, base, m = where['mtp_dspark.' + name]
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


def unrecord(rec):
    b = rec.view(np.uint16).reshape(rec.shape[0], rec.shape[1], 256)
    t = np.zeros((rec.shape[0], rec.shape[1], 16, 16), np.uint16)
    t[:, :, GV[:, 0], GV[:, 1]] = b
    return t.transpose(0, 2, 1, 3).reshape(rec.shape[0] * 16, rec.shape[1] * 16)


man = json.load(open(os.path.join(out, f'rank{r}.manifest.json')))
pk = open(os.path.join(out, f'rank{r}.pack'), 'rb')


def got(e):
    pk.seek(e['offset'])
    a = np.frombuffer(pk.read(e['nbytes']), np.uint8 if e['dtype'] == 'u8' else np.uint16)
    return a.reshape(e['shape'])


HR, HD = 64 // NR, 128
T = {e['name']: e for e in man['tensors']}
want = {'DS.tp.gv': lambda: src('backbone.target_projection.weight')[1536 * r:1536 * (r + 1)],
        'DS.target_norm': lambda: src('backbone.target_norm.weight'), 'DS.final_norm': lambda: src('backbone.final_norm.weight'),
        'DS.mask': lambda: src('backbone.mask_embedding'),
        'DS.jp.gv': lambda: src('autoregressive_head.joint_proj.weight'), 'DS.jp_bias': lambda: src('autoregressive_head.joint_proj.bias'),
        'DS.ar_out.gv': lambda: src('autoregressive_head.output.weight')[51200 * r:51200 * (r + 1)],
        'DS.ar_emb': lambda: src('autoregressive_head.embedding.weight')}
for j in range(3):
    L = f'backbone.layers.{j}.'
    want.update({
        f'DS.L{j}.wq.gv': lambda L=L: src(L + 'attention.wq.weight')[HR * HD * r:HR * HD * (r + 1)],
        f'DS.L{j}.kva.gv': lambda L=L: src(L + 'attention.wkv_a_with_mqa.weight'),
        f'DS.L{j}.kvb.gv': lambda L=L: src(L + 'attention.wkv_b.weight')[HR * 192 * r:HR * 192 * (r + 1)],
        f'DS.L{j}.wo.gv': lambda L=L: src(L + 'attention.wo.weight')[:, HR * HD * r:HR * HD * (r + 1)],
        f'DS.L{j}.w1.gv': lambda L=L: src(L + 'feed_forward.w1.weight')[6144 * r:6144 * (r + 1)],
        f'DS.L{j}.w3.gv': lambda L=L: src(L + 'feed_forward.w3.weight')[6144 * r:6144 * (r + 1)],
        f'DS.L{j}.w2.gv': lambda L=L: src(L + 'feed_forward.w2.weight')[:, 6144 * r:6144 * (r + 1)],
        f'DS.L{j}.attn_norm': lambda L=L: src(L + 'attention_norm.weight'), f'DS.L{j}.ffn_norm': lambda L=L: src(L + 'ffn_norm.weight'),
        f'DS.L{j}.kv_norm': lambda L=L: src(L + 'attention.kv_a_norm.weight')})
bad = 0
for name, f in want.items():
    e = T.get(name)
    if e is None:
        print('MISSING', name); bad += 1; continue
    g = got(e)
    if name.endswith('.gv'):
        g = unrecord(g)
    w = f()
    if g.shape != w.shape or not np.array_equal(g, w):
        print('MISMATCH', name, g.shape, w.shape); bad += 1
extra = set(T) - set(want)
print(f'rank {r}: {len(want)} tensors checked, {bad} bad, {len(extra)} unexpected' + (f' {sorted(extra)}' if extra else ''))
sys.exit(1 if bad or extra else 0)
