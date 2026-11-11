#!/usr/bin/env python3
"""The Mistral Large 4 EXL3 3.0 bpw pack -> gr8 rank packs, no requantization.

Per layer, rank r of NR (heads of layer l: chunk (r - l) mod NR of 64, 16 each at TP4, as tools/convert/plan.py):
  Lnnn.attn_norm, Lnnn.mlp_norm           bf16 [6144]        attention_norm, ffn_norm (stored BF16)
  Lnnn.kv_a_norm                          bf16 [512]         owners of a latent only (layers with wkv_a)
  Lnnn.kv_a.f16.gv                        u8 [36][384][512]  F16W record of wkv_a rows 0-575 (NoPE layers: 512 rows +
                                                             64 zero rows, so their rope key is 0); owners only
  Lnnn.kv_b.uk                            bf16 [16][192][512] the rank's heads' UK rows (64 rope / 128 NoPE), zero rows to 192
  Lnnn.kv_b.uv                            bf16 [16][256][512] the rank's heads' UV rows (128), zero rows to 256
  Lnnn.ssmax                              f32 [2][64]        attention ssmax s, b (all heads)
  Lnnn.q_b.{trellis,suh,svh}              dtr side 1         wq, the rank's 16 heads x 128 output columns
  Lnnn.o_proj.{trellis,suh,svh}           dtr side 2         wo, the rank's 16 x 128 input rows
  full-index layers: Lnnn.idx.q_norm      f32 [2][6144]      q_input_norm weight, bias
                     Lnnn.idx.wq_b.f16.gv u8 [256][384][512], Lnnn.idx.wk.f16.gv [8][384][512],
                     Lnnn.idx.weights_proj.f16.gv [2][384][512]
                     Lnnn.idx.k_norm.w / .b bf16 [128], Lnnn.idx.ssmax f32 [2][32]
  layer 0:           Lnnn.dense.{gate,up,down}.{trellis,suh,svh}   dtr, the rank's 6144 channels (chunk (r - l) mod NR)
  MoE layers:        Lnnn.router.f16.gv u8 [8][384][512]; Lnnn.tr3.* (gr8's tr3 expert layout, the shared expert
                     as row 128: E = 129, expert_format tr3-mul1)
Globals: G.rope_freq f32 [32] (LeRoPE x YaRN, from transformers), G.embed bf16 [204800][6144], G.final_norm bf16 [6144],
G.lm_head.f16.gv u8 [VN/16][384][512] (the rank's
vocab rows of the 6-bit head dequantized to fp16 = ExLlamaV3's weight W_eff; dtr cannot take K6).
F16 tensors that go to bf16 are checked bit-exact (they are bf16 originals).

Usage: ml4_pack.py PACK_DIR GR8_SRC EXL3_REF_DIR NR OUT_DIR --layers 0,1,3,35,39 [--ranks 0,1,2,3] [--no-globals]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

H, NHEAD, NB, NEXP, KSH, VOCAB, IDX_HEADS = 6144, 64, 48, 128, 384, 204800, 32
ap = argparse.ArgumentParser()
ap.add_argument('pack')
ap.add_argument('gr8')
ap.add_argument('exl3_ref')
ap.add_argument('nr', type=int)
ap.add_argument('out')
ap.add_argument('--layers', required=True)
ap.add_argument('--ranks', default='')
ap.add_argument('--no-globals', action='store_true')
a = ap.parse_args()
sys.path.insert(0, a.gr8)
sys.path.insert(0, a.exl3_ref)
from tools.convert.pack import PackWriter  # noqa: E402

NR = a.nr
ranks = [int(r) for r in a.ranks.split(',')] if a.ranks else list(range(NR))
layers = [int(x) for x in a.layers.split(',')]
cfg = json.load(open(os.path.join(a.pack, 'config.json')))
tc = cfg.get('text_config', cfg)
index = json.load(open(os.path.join(a.pack, 'model.safetensors.index.json')))['weight_map']
DT = {'I16': np.int16, 'F16': np.float16, 'I32': np.int32, 'BF16': np.uint16}


def load(name):
    with open(os.path.join(a.pack, index[name]), 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        m = json.loads(f.read(n))[name]
        s, e = m['data_offsets']
        f.seek(8 + n + s)
        return np.frombuffer(f.read(e - s), DT[m['dtype']]).reshape(m['shape'])


def has(name):
    return name in index


def bf16_of(x):
    """F16 -> bf16 bits, refusing any value bf16 cannot hold exactly; BF16 (uint16 bits) passes through."""
    if x.dtype == np.uint16:
        return x
    f = x.astype(np.float32)
    b = f.view(np.uint32)
    if np.any(b & 0xffff):
        raise SystemExit(f'F16 -> bf16 not exact for {int(np.count_nonzero(b & 0xffff))} values')
    return (b >> 16).astype(np.uint16)


def f32_of(x):
    return (x.astype(np.uint32) << 16).view(np.float32) if x.dtype == np.uint16 else x.astype(np.float32)


GV = []
for l_ in range(32):
    r_, c_ = l_ // 4, l_ % 4
    GV += [(r_, 2 * c_), (r_, 2 * c_ + 1), (r_ + 8, 2 * c_), (r_ + 8, 2 * c_ + 1),
           (r_, 2 * c_ + 8), (r_, 2 * c_ + 9), (r_ + 8, 2 * c_ + 8), (r_ + 8, 2 * c_ + 9)]
GV = np.array(GV)


def f16_record(w):
    """fp16 [O][K] -> u8 [O/16][K/16][512], the BF16W / F16W record order (loader-rs records.rs bf16_record)."""
    w = np.ascontiguousarray(w, np.float16)
    O, K = w.shape
    assert O % 16 == 0 and K % 16 == 0, w.shape
    t = w.reshape(O // 16, 16, K // 16, 16).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t[:, :, GV[:, 0], GV[:, 1]]).view(np.uint8)


def chunk(n, l, r):
    """(start, count) of rank r's chunk of n units on layer l: chunk (r - l) mod NR, the larger chunks first."""
    w = [n // NR + (1 if j < n % NR else 0) for j in range(NR)]
    j = (r - l) % NR
    return sum(w[:j]), w[j]


def trellis(prefix):
    if int(load(prefix + '.mul1').reshape(-1)[0]) & 0xffffffff != 0x83DCD12D:
        raise SystemExit(f'{prefix}: not the mul1 codebook')
    return load(prefix + '.trellis'), load(prefix + '.suh'), load(prefix + '.svh')


def dtr_side1(prefix, c0, c1):
    """output columns c0..c1 (128-aligned): entries [E][k16][8][16K] = trellis[:, 8e:8e+8]"""
    tr, suh, svh = trellis(prefix)
    ent = np.stack([tr[:, 8 * e:8 * e + 8] for e in range(c0 // 128, c1 // 128)])
    return ent.reshape(ent.shape[0], -1).view(np.uint8), suh, svh[c0:c1]


def dtr_side2(prefix, r0, r1):
    """input rows r0..r1 (128-aligned): entries [E][b][r][c][16K] = trellis[8e+r, 8b+c]"""
    tr, suh, svh = trellis(prefix)
    KT, NT, W = tr.shape
    ent = np.stack([tr[8 * e:8 * e + 8].reshape(8, NT // 8, 8, W).transpose(1, 0, 2, 3) for e in range(r0 // 128, r1 // 128)])
    return ent.reshape(ent.shape[0], -1).view(np.uint8), suh[r0:r1], svh


meta = dict(converter='ml4_pack', model='mistral_large4', nr=NR, layers=layers,
            source_pack=os.path.basename(os.path.normpath(a.pack)), expert_formats={str(l): 'tr3-mul1' for l in layers if l >= 1})
os.makedirs(a.out, exist_ok=True)
W = {r: PackWriter(os.path.join(a.out, f'rank{r}.pack'), r, False, meta) for r in ranks}
xf = dict(expert_format='tr3-mul1', expert_source='exl3-mul1 (Mistral-Large-4 EXL3 3.0 bpw pack)')


def put_dtr(w, name, ent, suh, svh, side):
    w.put(name + '.trellis', ent, 'u8', layout=f'dtr side {side} entries [E][bytes] (EXL3 K3 mul1 tiles as stored)')
    w.put(name + '.suh', np.ascontiguousarray(suh, np.float16), 'f16')
    w.put(name + '.svh', np.ascontiguousarray(svh, np.float16), 'f16')


for l in layers:
    P, S = f'L{l:03d}', f'layers.{l}.'
    rope = tc['no_rope_layers'][l] == 1
    own_kv = tc['kv_source_layers'][l] == l
    full = tc['indexer_types'][l] == 'full'
    nope = 64 if rope else 128
    kvb = load(S + 'attention.wkv_b.weight').reshape(NHEAD, nope + 128, 512)
    ss = np.stack([load(S + 'attention.ssmax_scaling.s'), load(S + 'attention.ssmax_scaling.b')]).astype(np.float32)
    shared = {}
    if own_kv:
        kva = load(S + 'attention.wkv_a_with_mqa.weight')
        kva = np.concatenate([kva[:576], np.zeros((576 - kva[:576].shape[0], H), np.float16)]) if kva.shape[0] < 576 else kva[:576]
        assert kva.shape[0] == 576 and (rope or not kva[512:].any())
        shared['kv_a'] = f16_record(kva)
        shared['kv_a_norm'] = load(S + 'attention.kv_a_norm.weight')
    if full:
        shared['q_norm'] = np.stack([load(S + 'attention.indexer.q_input_norm.weight'), load(S + 'attention.indexer.q_input_norm.bias')]).astype(np.float32)
        shared['wq_b'] = f16_record(load(S + 'attention.indexer.wq_b.weight'))
        shared['wk'] = f16_record(load(S + 'attention.indexer.wk.weight'))
        shared['wp'] = f16_record(load(S + 'attention.indexer.weights_proj.weight'))
        shared['k_norm'] = (bf16_of(load(S + 'attention.indexer.k_norm.weight')), bf16_of(load(S + 'attention.indexer.k_norm.bias')))
        shared['iss'] = np.stack([load(S + 'attention.indexer.ssmax_scaling.s'), load(S + 'attention.indexer.ssmax_scaling.b')]).astype(np.float32)
    if l >= 1:
        shared['router'] = f16_record(load(S + 'gate.weight'))
    for r, w in W.items():
        h0, nh = chunk(NHEAD, l, r)
        w.put(P + '.attn_norm', bf16_of(load(S + 'attention_norm.weight')), 'bf16')
        w.put(P + '.mlp_norm', bf16_of(load(S + 'ffn_norm.weight')), 'bf16')
        if own_kv:
            w.put(P + '.kv_a_norm', bf16_of(shared['kv_a_norm']), 'bf16')
            w.put(P + '.kv_a.f16.gv', shared['kv_a'], 'u8', layout='F16W record [O/16][K/16][512], 576 rows')
        uk = np.zeros((nh, 192, 512), np.uint16)
        uv = np.zeros((nh, 256, 512), np.uint16)
        uk[:, :nope] = bf16_of(kvb[h0:h0 + nh, :nope])
        uv[:, :128] = bf16_of(kvb[h0:h0 + nh, nope:])
        w.put(P + '.kv_b.uk', uk, 'bf16', layout=f'[h][192][512] rows 0-{nope - 1} UK, rest zero (heads {h0}-{h0 + nh - 1})')
        w.put(P + '.kv_b.uv', uv, 'bf16', layout='[h][256][512] rows 0-127 UV, rest zero')
        w.put(P + '.ssmax', ss, 'f32', layout='[2][64] s, b')
        put_dtr(w, P + '.q_b', *dtr_side1(S + 'attention.wq', 128 * h0, 128 * (h0 + nh)), 1)
        put_dtr(w, P + '.o_proj', *dtr_side2(S + 'attention.wo', 128 * h0, 128 * (h0 + nh)), 2)
        if full:
            w.put(P + '.idx.q_norm', shared['q_norm'], 'f32', layout='[2][6144] LayerNorm weight, bias (eps 1e-6)')
            w.put(P + '.idx.wq_b.f16.gv', shared['wq_b'], 'u8')
            w.put(P + '.idx.wk.f16.gv', shared['wk'], 'u8')
            w.put(P + '.idx.weights_proj.f16.gv', shared['wp'], 'u8')
            w.put(P + '.idx.k_norm.w', shared['k_norm'][0], 'bf16')
            w.put(P + '.idx.k_norm.b', shared['k_norm'][1], 'bf16')
            w.put(P + '.idx.ssmax', shared['iss'], 'f32', layout='[2][32] s, b')
        if l == 0:
            c0, nc = chunk(tc['intermediate_size'] // 128, l, r)
            c0, c1 = 128 * c0, 128 * (c0 + nc)
            put_dtr(w, P + '.dense.gate', *dtr_side1(S + 'feed_forward.w1', c0, c1), 1)
            put_dtr(w, P + '.dense.up', *dtr_side1(S + 'feed_forward.w3', c0, c1), 1)
            put_dtr(w, P + '.dense.down', *dtr_side2(S + 'feed_forward.w2', c0, c1), 2)
        else:
            w.put(P + '.router.f16.gv', shared['router'], 'u8', layout='F16W record, 128 experts')
    if l >= 1:   # routed experts + the shared expert as row 128 (gr8's tr3 layout, every rank at once)
        rows = [(e, f'{S}experts.{e}') for e in range(NEXP)] + [(NEXP, f'{S}shared_experts')]
        E = len(rows)
        own = {r: {e: [b for b in range(NB) if (48 * e + b + l) % NR == r] for e, _ in rows} for r in ranks}
        acc = {r: dict(fc1=[], fc2=[], v1=[], v2=[], v3=[], table=np.zeros((E, 6), np.int64), o1=0, o2=0, row=0) for r in ranks}
        for i, (e, prefix) in enumerate(rows):
            (tg, sug, svg), (tu, suu, svu), (td, sud, svd) = (trellis(f'{prefix}.{m}') for m in ('w1', 'w3', 'w2'))
            k = tg.shape[-1] // 16
            half = KSH * 8 * 16 * k * 2
            for r in ranks:
                A, bs = acc[r], own[r][e]
                A['table'][i] = (A['o1'], A['o2'], A['row'], len(bs), k, bs[0] if bs else 0)
                for b in bs:
                    A['fc1'] += [np.ascontiguousarray(tg[:, 8 * b:8 * b + 8]), np.ascontiguousarray(tu[:, 8 * b:8 * b + 8])]
                    A['fc2'].append(np.ascontiguousarray(td[8 * b:8 * b + 8].transpose(1, 0, 2)))
                    A['v1'].append(np.stack([sug, suu]))
                    A['v2'].append(np.stack([svg[128 * b:128 * b + 128], svu[128 * b:128 * b + 128], sud[128 * b:128 * b + 128]]))
                    A['v3'].append(svd)
                    A['o1'] += 2 * half
                    A['o2'] += half
                    A['row'] += 1
        for r, w in W.items():
            A = acc[r]
            for name, nbytes in (('fc1', A['o1']), ('fc2', A['o2'])):
                w.begin(f'{P}.tr3.{name}', 'u8', (nbytes,), **xf)
                for p in A[name]:
                    w.write(p.view(np.uint8))
                w.end()
            w.put(f'{P}.tr3.v1', np.stack(A['v1']).astype(np.float16), 'f16', **xf)
            w.put(f'{P}.tr3.v2', np.stack(A['v2']).astype(np.float16), 'f16', **xf)
            w.put(f'{P}.tr3.v3', np.stack(A['v3']).astype(np.float16), 'f16', **xf)
            w.put(f'{P}.tr3.table', A['table'], 'i64', **xf)
            w.put(f'{P}.tr3.kmix', np.zeros(E, np.int64), 'i64', **xf)
            # K1 also reads an int4 shared-expert record (bind.cuh make_k1: a trellis shared expert is table row NEXP + 1,
            # and ML4's is a routed row, 128). A minimal zero record of 32 channels, as P3b's k1_ml4_test: output 0.
            w.put(f'{P}.shared.gptq.fc1', np.zeros((2, 192, 576), np.uint8), 'u8', layout='zero int4 record, 32 channels (unused)')
            w.put(f'{P}.shared.gptq.fc2', np.zeros((384, 1, 288), np.uint8), 'u8', layout='zero int4 record, 32 channels (unused)')
    print(f'{P}: rope {rope}, owns kv {own_kv}, indexer {"full" if full else "shared"}, {"dense" if l == 0 else "moe"}', flush=True)

if not a.no_globals:
    import torch
    import exl3_ref
    emb = load('tok_embeddings.weight')
    tr, suh, svh = trellis('output')
    VN = VOCAB // NR
    # LeRoPE x YaRN frequencies, from transformers' own MistralLarge4TextRotaryEmbedding (its forward: inv_freq.float() *
    # alpha.float().exp(), fp32), so the engine's angle pos * f_j is the reference's (attn.cu cs, g6a_ml4_rope)
    from transformers import AutoConfig
    from transformers.models.mistral_large4 import modeling_mistral_large4 as mm
    rcfg = AutoConfig.from_pretrained(a.pack)
    rot = mm.MistralLarge4TextRotaryEmbedding(rcfg.text_config if hasattr(rcfg, 'text_config') else rcfg)
    alpha = torch.from_numpy(f32_of(load('lerope.alpha')))
    freq = (rot.inv_freq.float() * alpha.float().exp()).numpy().astype(np.float32)
    assert freq.shape == (32,) and rot.attention_scaling == 1.0, (freq.shape, rot.attention_scaling)
    for r, w in W.items():
        w.put('G.rope_freq', freq, 'f32', layout='[32] f_j = YaRN inv_freq_j * exp(lerope.alpha_j), fp32 (transformers 5.22.2)')
        w.put('G.embed', emb, 'bf16')
        w.put('G.final_norm', bf16_of(load('norm.weight')), 'bf16')
        v0 = VN * r
        weff = exl3_ref.dequant_exl3(torch.from_numpy(tr[:, v0 // 16:(v0 + VN) // 16].copy()), torch.from_numpy(suh.copy()),
                                     torch.from_numpy(svh[v0:v0 + VN].copy()), mul1=0x83DCD12D)   # [6144][VN] fp32
        w.put('G.lm_head.f16.gv', f16_record(weff.T.contiguous().half().numpy()), 'u8',
              layout=f'F16W record of vocab rows {v0}-{v0 + VN - 1}: the 6-bit head dequantized to fp16 (ExLlamaV3 W_eff)')
        print(f'rank {r}: globals', flush=True)
for r, w in W.items():
    m = w.close()
    print(f'rank {r}: {m["pack_bytes"] / 1e9:.2f} GB', flush=True)
