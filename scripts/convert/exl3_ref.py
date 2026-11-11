#!/usr/bin/env python3
"""Pure-PyTorch (CPU) reference decoder for EXL3 trellis-quantized linear layers.

Derived from (no guessing):
  * cuda_exl3/csrc/gemm.cu, hadamard.cu, linear.py, ops.py  (plugin, MIT)
  * ExLlamaV3 exllamav3_ext/quant/{exl3_dq.cuh, codebook.cuh, reconstruct.cu,
    hadamard_inner.cuh} and modules/quant/exl3.py:get_weight_tensor (the plugin's gemm.cu
    includes exl3_dq.cuh / exl3_had.cuh from there: "part of the on-disk format").

FORMAT
  trellis : int16 [K/16, N/16, 16*bits]    K = in features, N = out features
            one 16x16 tile = 16*bits int16 = 8*bits uint32 words = a cyclic bit stream of
            256*bits bits.  Words are loaded as little-endian uint32 from the int16 array, the
            bit stream is MSB-first inside each uint32, words in increasing order.
  state_p : for tile position p (0..255), the 16-bit window that ENDS at stream bit
            (p+1)*bits (cyclic, mod 256*bits), MSB first.  (sliding K-bit window, tail-biting)
  value_p = codebook(state_p)   (fp16 value; mul1: (1024 + bytesum(state*0x83DCD12D)) * k_inv + k_bias)
  tile (k,n) <- position p = 8*(4*(n%8) + (k%8)//2) + 4*(n//8) + 2*(k//8) + (k%2)
            (= the mma.m16n8k16 B-fragment layout: thread lane holds positions 8*lane..8*lane+7,
             half2 pairs consecutive k; lane = 4*n_in_8 + k_in_8//2; second n8 half = +4)
  W_inner[K,N] = stack of tiles, tile (kt,nt) at rows 16kt.., cols 16nt..   (stored [in, out]!)
  y = svh * H_out( H_in(x * suh) @ W_inner )      H = Sylvester Hadamard / sqrt(128) on
                                                  independent 128-element blocks
  => W_eff[in,out] = diag(suh) . blockdiag(H128) . W_inner . blockdiag(H128) . diag(svh),   y = x @ W_eff
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

TILE = 16
HAD = 128
CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
MUL1_MULT = 0x83DCD12D          # == 2212286765 == (int32) -2082680531, the "mul1" marker tensor value
MCG_MULT = 0xCBAC1FED


# ----------------------------------------------------------------------------- codebooks
_LUT: dict[int, torch.Tensor] = {}


def codebook_lut(cb: int) -> torch.Tensor:
    """float32[65536]: fp16 value the CUDA kernel produces for every 16-bit trellis state."""
    if cb in _LUT:
        return _LUT[cb]
    s = np.arange(65536, dtype=np.uint32)
    if cb == CB_MUL1:
        x = s * np.uint32(MUL1_MULT)                                  # u32 wrap-around
        bsum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)   # dp4a(x, 0x01010101, 0)
        h = (np.uint32(0x6400) + bsum).astype(np.uint16).view(np.float16).astype(np.float64)  # 1024.0 + bsum
        k_inv = np.array([0x1EEE], np.uint16).view(np.float16)[0].astype(np.float64)
        k_bias = np.array([0xC931], np.uint16).view(np.float16)[0].astype(np.float64)
        out = (h * k_inv + k_bias).astype(np.float16)                 # __hfma: one rounding
    elif cb in (CB_MCG, CB_3INST):
        x = s * np.uint32(MCG_MULT if cb == CB_MCG else 89226354)
        if cb == CB_3INST:
            x = x + np.uint32(64248484)
        x = (x & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)       # lop3 0x6a
        lo = (x & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
        hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
        out = (lo + hi).astype(np.float16)                            # __hadd
    else:
        raise ValueError(f"unknown codebook {cb}")
    t = torch.from_numpy(out.astype(np.float32))
    _LUT[cb] = t
    return t


def codebook_from_multiplier(m) -> int:
    """Map the checkpoint's `mul1` / `mcg` int32 scalar (or None) to a codebook id."""
    if isinstance(m, torch.Tensor):
        m = int(m.reshape(-1)[0].item())
    m = int(m) & 0xFFFFFFFF
    if m == MUL1_MULT:
        return CB_MUL1
    if m == MCG_MULT:
        return CB_MCG
    raise ValueError(f"multiplier {m:#x} is neither mul1 ({MUL1_MULT:#x}) nor mcg ({MCG_MULT:#x})")


# ----------------------------------------------------------------------------- trellis unpack
def _window_index(bits: int):
    """Chunk index a[p] (16-bit chunks of the swapped stream) and bit offset o[p] of window p."""
    n_bits = 256 * bits
    p = np.arange(256)
    start = ((p + 1) * bits - 16) % n_bits
    return torch.from_numpy(start // 16), torch.from_numpy(start % 16), n_bits // 16


def unpack_states(trellis: torch.Tensor, bits: int | None = None) -> torch.Tensor:
    """int16 [KT, NT, 16*bits] -> int64 states [KT, NT, 256] (tile position order)."""
    KT, NT, W = trellis.shape
    bits = W // 16 if bits is None else bits
    assert W == 16 * bits, (trellis.shape, bits)
    a, o, C = _window_index(bits)
    c = (trellis.reshape(KT * NT, W).to(torch.int64)) & 0xFFFF            # u16 chunks, memory order
    c = c.reshape(-1, C // 2, 2).flip(-1).reshape(-1, C)                  # LE uint32 -> MSB-first 16-bit chunks
    hi = c[:, a]
    lo = c[:, (a + 1) % C]
    st = (((hi << 16) | lo) >> (16 - o)) & 0xFFFF
    return st.reshape(KT, NT, 256)


def unpack_states_literal(trellis_tile: torch.Tensor, bits: int) -> list[int]:
    """Literal python port of exl3_dq.cuh::dq() (single value) for one tile; test oracle."""
    u16 = [int(v) & 0xFFFF for v in trellis_tile.reshape(-1).tolist()]
    ptr = [u16[2 * j] | (u16[2 * j + 1] << 16) for j in range(8 * bits)]
    out = []
    for t in range(256):
        b0 = t * bits + bits - 16 + 256 * bits
        b1 = b0 + 16
        i0 = b0 // 32
        i1 = (b1 - 1) // 32
        s0 = (i1 + 1) * 32 - b1
        a = ptr[i0 % (bits * 256 // 32)]
        b = ptr[i1 % (bits * 256 // 32)]
        out.append((((a << 32) | b) >> s0) & 0xFFFF)                     # __funnelshift_r(b, a, s0)
    return out


def _tile_perm() -> torch.Tensor:
    k = torch.arange(16).view(16, 1)
    n = torch.arange(16).view(1, 16)
    return 8 * (4 * (n % 8) + (k % 8) // 2) + 4 * (n // 8) + 2 * (k // 8) + (k % 2)   # [16(k),16(n)] -> p


_PERM = _tile_perm()


def dequant_inner(trellis: torch.Tensor, cb: int, bits: int | None = None) -> torch.Tensor:
    """Trellis -> W_inner[K, N] float32 (fp16-valued), i.e. what the tensor-core B operand holds."""
    KT, NT, _ = trellis.shape
    st = unpack_states(trellis, bits)
    vals = codebook_lut(cb)[st]                                           # [KT, NT, 256]
    tiles = vals[..., _PERM]                                              # [KT, NT, 16(k), 16(n)]
    return tiles.permute(0, 2, 1, 3).reshape(KT * TILE, NT * TILE).contiguous()


# ----------------------------------------------------------------------------- hadamard
_H: dict[int, torch.Tensor] = {}


def had128() -> torch.Tensor:
    """Sylvester (natural order) Hadamard / sqrt(128), float32, symmetric and involutory."""
    if 0 not in _H:
        h = torch.ones(1, 1, dtype=torch.float64)
        for _ in range(7):
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[0] = (h / math.sqrt(HAD)).float()
    return _H[0]


def had_blocks(x: torch.Tensor, dim: int = -1, block: int = HAD) -> torch.Tensor:
    """Independent block-diagonal Hadamard over `block`-sized chunks of `dim` (block must be 128)."""
    assert block == HAD
    x = x.movedim(dim, -1)
    sh = x.shape
    assert sh[-1] % HAD == 0, f"dim {sh[-1]} is not a multiple of {HAD}"
    y = x.reshape(*sh[:-1], sh[-1] // HAD, HAD) @ had128().to(x.dtype)
    return y.reshape(sh).movedim(-1, dim)


# ----------------------------------------------------------------------------- dequant / linear
def _resolve_cb(mul1, cb):
    if cb is not None:
        return cb
    if mul1 is None:
        raise ValueError("pass the mul1/mcg marker tensor (or multiplier) or cb=")
    return codebook_from_multiplier(mul1)


def iter_dequant_blocks(trellis, suh, svh, mul1=None, cb=None, bits=None):
    """Yield (row0, W_eff[128, N] float32) for each 128-row input block (low memory)."""
    cb = _resolve_cb(mul1, cb)
    KT, NT, _ = trellis.shape
    K, N = KT * TILE, NT * TILE
    assert K % HAD == 0 and N % HAD == 0, (K, N)
    suh = suh.float().reshape(-1)
    svh = svh.float().reshape(-1)
    assert suh.numel() == K and svh.numel() == N, (suh.shape, svh.shape, K, N)
    H = had128()
    for r0 in range(0, K, HAD):
        inner = dequant_inner(trellis[r0 // TILE:(r0 + HAD) // TILE], cb, bits)       # [128, N]
        blk = H @ inner                                                               # input Hadamard (rows)
        blk = blk * suh[r0:r0 + HAD, None]                                            # suh scales input rows
        blk = (blk.reshape(HAD, N // HAD, HAD) @ H).reshape(HAD, N)                   # output Hadamard (cols)
        yield r0, blk * svh[None, :]                                                  # svh scales output cols


def dequant_exl3(trellis, suh, svh, mul1=None, cb=None, bits=None) -> torch.Tensor:
    """W_eff[in, out] float32 with  y = x @ W_eff  ==  exl3_linear(x, trellis, suh, svh)."""
    KT, NT, _ = trellis.shape
    W = torch.empty(KT * TILE, NT * TILE, dtype=torch.float32)
    for r0, blk in iter_dequant_blocks(trellis, suh, svh, mul1, cb, bits):
        W[r0:r0 + HAD] = blk
    return W


def _f16(t, on):
    return t.half().float() if on else t


def exl3_linear_ref(x, trellis, suh, svh, mul1=None, cb=None, bits=None,
                    group_n=None, emulate_fp16=False) -> torch.Tensor:
    """out = had(x * suh) @ dequant(trellis) -> had -> * svh, staged exactly like the CUDA kernels.

    x       [..., K]
    trellis [K/16, N/16, 16*bits] int16
    suh     [K] or [groups, K] (one row per fused shard; shard g uses output columns
            [sum(group_n[:g]), sum(group_n[:g+1])) )
    svh     [N]
    emulate_fp16: round A_had, the accumulator result and the output to fp16 at the same points
            the kernels do (x*suh is formed in fp32 here; the kernel multiplies in float too).
    """
    cb = _resolve_cb(mul1, cb)
    KT, NT, _ = trellis.shape
    K, N = KT * TILE, NT * TILE
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K).float()
    m = x2.shape[0]
    suh = suh.float().reshape(-1, K)
    if group_n is None:
        group_n = [N]
    assert sum(group_n) == N and suh.shape[0] == len(group_n), (group_n, suh.shape, N)
    H = had128()
    acc = torch.zeros(m, N)
    for r0 in range(0, K, HAD):
        inner = dequant_inner(trellis[r0 // TILE:(r0 + HAD) // TILE], cb, bits)     # [128, N]
        off = 0
        for g, ng in enumerate(group_n):
            a = _f16((x2[:, r0:r0 + HAD] * suh[g, r0:r0 + HAD]) @ H, emulate_fp16)  # A_had (fp16 in kernel)
            acc[:, off:off + ng] += a @ inner[:, off:off + ng]                       # fp32 accumulate
            off += ng
    c = _f16(acc, emulate_fp16)                                                      # accumulator -> half
    y = (c.reshape(m, N // HAD, HAD) @ H).reshape(m, N) * svh.float().reshape(1, N)  # output had, then svh
    y = _f16(y, emulate_fp16)
    return y.reshape(*lead, N)


# ----------------------------------------------------------------------------- self test
def selftest():
    g = torch.Generator().manual_seed(1)
    # 1) vectorised bit-window unpack == literal port of the CUDA dq()
    for bits in range(1, 9):
        tr = torch.randint(-32768, 32768, (3, 2, 16 * bits), generator=g, dtype=torch.int16)
        st = unpack_states(tr, bits)
        for kt in range(3):
            for nt in range(2):
                assert st[kt, nt].tolist() == unpack_states_literal(tr[kt, nt], bits), (bits, kt, nt)
    print("unpack_states == literal dq() port for bits 1..8: OK")
    # 2) codebooks sane (unit-ish Gaussian, finite)
    for cb, nm in ((CB_MUL1, "mul1"), (CB_MCG, "mcg"), (CB_3INST, "3inst")):
        l = codebook_lut(cb)
        assert torch.isfinite(l).all()
        print(f"codebook {nm}: mean {l.mean():+.4f} std {l.std():.4f} min {l.min():+.3f} max {l.max():+.3f}")
    # 3) tile permutation is a bijection
    assert sorted(_PERM.reshape(-1).tolist()) == list(range(256))
    # 4) Hadamard orthonormal & involutory
    H = had128()
    assert torch.allclose(H @ H, torch.eye(HAD), atol=1e-5) and torch.allclose(H, H.T)
    # 5) staged linear == x @ W_eff
    for bits in (3, 4, 6, 8):
        tr = torch.randint(-32768, 32768, (16, 8, 16 * bits), generator=g, dtype=torch.int16)
        suh = torch.randn(256, generator=g).half()
        svh = torch.randn(128, generator=g).half()
        x = torch.randn(5, 256, generator=g)
        for cb in (CB_MUL1, CB_MCG):
            y1 = exl3_linear_ref(x, tr, suh, svh, cb=cb)
            y2 = x @ dequant_exl3(tr, suh, svh, cb=cb)
            rel = (y1 - y2).norm() / y2.norm()
            assert rel < 1e-5, (bits, cb, rel)
    print("exl3_linear_ref == x @ dequant_exl3 (bits 3,4,6,8; mul1,mcg): OK")


if __name__ == "__main__":
    selftest()
