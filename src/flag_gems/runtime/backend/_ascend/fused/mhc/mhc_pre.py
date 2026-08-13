# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ascend implementation of mHC Pre operator.

Public API matches flag_gems.fused.mhc.mhc_pre:

    mhc_pre(residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, n_splits=1)
        -> (post_mix, comb_mix, layer_input)

Internally uses Ascend-optimised Triton kernels:
- num_stages=1 throughout (NPU does not benefit from multi-stage pipelining).
- BLOCK_T vectorised Sinkhorn loops: keeps 16 comb scalars in registers
  across all Sinkhorn iterations.
- torch.mm for the heavy GEMM (matches production MMAD on Ascend).

Note: hc_post_mult_value is expected to be 2.0 (the aclnn convention used
throughout this codebase). If another value is supplied a warning is emitted
and the call falls back to the PyTorch reference.
"""

from __future__ import annotations

import logging
import weakref

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# bf16 weight cache (inference fast-path)
# ---------------------------------------------------------------------------

_FN_BF16_CACHE: weakref.WeakKeyDictionary[torch.Tensor, tuple[int, torch.Tensor]] = (
    weakref.WeakKeyDictionary()
)


def _get_fn_bf16_cached(fn: torch.Tensor) -> torch.Tensor:
    if fn.requires_grad or torch.is_grad_enabled():
        return fn.to(dtype=torch.bfloat16)
    version = fn._version
    cached = _FN_BF16_CACHE.get(fn)
    if cached is not None:
        cached_version, cached_bf16 = cached
        if cached_version == version:
            return cached_bf16
    fn_bf16 = fn.to(dtype=torch.bfloat16)
    _FN_BF16_CACHE[fn] = (version, fn_bf16)
    return fn_bf16


# ---------------------------------------------------------------------------
# RMS-norm kernel (inv_rms only; no x_scaled write)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["HC_D"],
)
@triton.jit
def _rms_only_kernel(
    x_ptr,
    inv_rms_ptr,
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,
    NORM_EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single-pass: compute inv_rms only (no x_scaled materialisation)."""
    pid = tl.program_id(0)
    base = pid * HC_D
    sq = 0.0
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sq += tl.sum(v * v, axis=0)
    inv = tl.math.rsqrt(sq * D_INV + NORM_EPS)
    tl.store(inv_rms_ptr + pid, inv)


# ---------------------------------------------------------------------------
# Fused sigmoid-heads + Sinkhorn + weighted-sum kernel (HC=4 fast path)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_heads_sinkhorn_yscale_hc4(
    mixes_ptr,    # (T, hc_mix),      float32 – already scaled by inv_rms
    inv_rms_ptr,  # (T,),             float32
    alpha_ptr,    # (3,),             float32
    base_ptr,     # (hc_mix,),        float32 — hc_mix = 4*(4+2) = 24
    x_ptr,        # (T, 4, D),        bfloat16
    pre_ptr,      # (T, 4),           float32  [out]
    post_ptr,     # (T, 4),           float32  [out]
    comb_ptr,     # (T, 4, 4),        float32  [out]
    y_ptr,        # (T, D),           bfloat16 [out]
    T,
    D,
    HC_EPS: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (cdiv(T, BLOCK_T),).
    Vectorised over BLOCK_T tokens; all 16 comb values kept in registers
    across all Sinkhorn iterations.  D-dimension loop handled inside.
    """
    pid = tl.program_id(0)
    t_off = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_off < T

    a0 = tl.load(alpha_ptr + 0)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)

    # ---- pre head (4 values) ----
    def _m(col):
        return tl.load(mixes_ptr + t_off * 24 + col, mask=t_mask, other=0.0)

    p0 = tl.sigmoid(_m(0) * a0 + tl.load(base_ptr + 0)) + HC_EPS
    p1 = tl.sigmoid(_m(1) * a0 + tl.load(base_ptr + 1)) + HC_EPS
    p2 = tl.sigmoid(_m(2) * a0 + tl.load(base_ptr + 2)) + HC_EPS
    p3 = tl.sigmoid(_m(3) * a0 + tl.load(base_ptr + 3)) + HC_EPS

    tl.store(pre_ptr + t_off * 4 + 0, p0, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 1, p1, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 2, p2, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 3, p3, mask=t_mask)

    # ---- post head (4 values) ----
    q0 = 2.0 * tl.sigmoid(_m(4) * a1 + tl.load(base_ptr + 4))
    q1 = 2.0 * tl.sigmoid(_m(5) * a1 + tl.load(base_ptr + 5))
    q2 = 2.0 * tl.sigmoid(_m(6) * a1 + tl.load(base_ptr + 6))
    q3 = 2.0 * tl.sigmoid(_m(7) * a1 + tl.load(base_ptr + 7))

    tl.store(post_ptr + t_off * 4 + 0, q0, mask=t_mask)
    tl.store(post_ptr + t_off * 4 + 1, q1, mask=t_mask)
    tl.store(post_ptr + t_off * 4 + 2, q2, mask=t_mask)
    tl.store(post_ptr + t_off * 4 + 3, q3, mask=t_mask)

    # ---- comb head: 4x4 logits -> softmax-row -> Sinkhorn (in-register) ----
    c00 = _m(8)  * a2 + tl.load(base_ptr + 8)
    c01 = _m(9)  * a2 + tl.load(base_ptr + 9)
    c02 = _m(10) * a2 + tl.load(base_ptr + 10)
    c03 = _m(11) * a2 + tl.load(base_ptr + 11)
    c10 = _m(12) * a2 + tl.load(base_ptr + 12)
    c11 = _m(13) * a2 + tl.load(base_ptr + 13)
    c12 = _m(14) * a2 + tl.load(base_ptr + 14)
    c13 = _m(15) * a2 + tl.load(base_ptr + 15)
    c20 = _m(16) * a2 + tl.load(base_ptr + 16)
    c21 = _m(17) * a2 + tl.load(base_ptr + 17)
    c22 = _m(18) * a2 + tl.load(base_ptr + 18)
    c23 = _m(19) * a2 + tl.load(base_ptr + 19)
    c30 = _m(20) * a2 + tl.load(base_ptr + 20)
    c31 = _m(21) * a2 + tl.load(base_ptr + 21)
    c32 = _m(22) * a2 + tl.load(base_ptr + 22)
    c33 = _m(23) * a2 + tl.load(base_ptr + 23)

    # row-wise softmax (row 0)
    rm0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    e00 = tl.exp(c00 - rm0); e01 = tl.exp(c01 - rm0)
    e02 = tl.exp(c02 - rm0); e03 = tl.exp(c03 - rm0)
    rs0 = e00 + e01 + e02 + e03
    c00 = e00 / rs0 + HC_EPS; c01 = e01 / rs0 + HC_EPS
    c02 = e02 / rs0 + HC_EPS; c03 = e03 / rs0 + HC_EPS

    # row 1
    rm1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    e10 = tl.exp(c10 - rm1); e11 = tl.exp(c11 - rm1)
    e12 = tl.exp(c12 - rm1); e13 = tl.exp(c13 - rm1)
    rs1 = e10 + e11 + e12 + e13
    c10 = e10 / rs1 + HC_EPS; c11 = e11 / rs1 + HC_EPS
    c12 = e12 / rs1 + HC_EPS; c13 = e13 / rs1 + HC_EPS

    # row 2
    rm2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    e20 = tl.exp(c20 - rm2); e21 = tl.exp(c21 - rm2)
    e22 = tl.exp(c22 - rm2); e23 = tl.exp(c23 - rm2)
    rs2 = e20 + e21 + e22 + e23
    c20 = e20 / rs2 + HC_EPS; c21 = e21 / rs2 + HC_EPS
    c22 = e22 / rs2 + HC_EPS; c23 = e23 / rs2 + HC_EPS

    # row 3
    rm3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))
    e30 = tl.exp(c30 - rm3); e31 = tl.exp(c31 - rm3)
    e32 = tl.exp(c32 - rm3); e33 = tl.exp(c33 - rm3)
    rs3 = e30 + e31 + e32 + e33
    c30 = e30 / rs3 + HC_EPS; c31 = e31 / rs3 + HC_EPS
    c32 = e32 / rs3 + HC_EPS; c33 = e33 / rs3 + HC_EPS

    # col normalise
    cs0 = c00 + c10 + c20 + c30
    cs1 = c01 + c11 + c21 + c31
    cs2 = c02 + c12 + c22 + c32
    cs3 = c03 + c13 + c23 + c33
    inv0 = 1.0 / (cs0 + HC_EPS); inv1 = 1.0 / (cs1 + HC_EPS)
    inv2 = 1.0 / (cs2 + HC_EPS); inv3 = 1.0 / (cs3 + HC_EPS)
    c00 *= inv0; c10 *= inv0; c20 *= inv0; c30 *= inv0
    c01 *= inv1; c11 *= inv1; c21 *= inv1; c31 *= inv1
    c02 *= inv2; c12 *= inv2; c22 *= inv2; c32 *= inv2
    c03 *= inv3; c13 *= inv3; c23 *= inv3; c33 *= inv3

    # remaining Sinkhorn iterations
    for _ in tl.static_range(SINKHORN_ITERS - 1):
        rs0 = c00 + c01 + c02 + c03
        rs1 = c10 + c11 + c12 + c13
        rs2 = c20 + c21 + c22 + c23
        rs3 = c30 + c31 + c32 + c33
        ir0 = 1.0 / (rs0 + HC_EPS); ir1 = 1.0 / (rs1 + HC_EPS)
        ir2 = 1.0 / (rs2 + HC_EPS); ir3 = 1.0 / (rs3 + HC_EPS)
        c00 *= ir0; c01 *= ir0; c02 *= ir0; c03 *= ir0
        c10 *= ir1; c11 *= ir1; c12 *= ir1; c13 *= ir1
        c20 *= ir2; c21 *= ir2; c22 *= ir2; c23 *= ir2
        c30 *= ir3; c31 *= ir3; c32 *= ir3; c33 *= ir3

        cs0 = c00 + c10 + c20 + c30
        cs1 = c01 + c11 + c21 + c31
        cs2 = c02 + c12 + c22 + c32
        cs3 = c03 + c13 + c23 + c33
        inv0 = 1.0 / (cs0 + HC_EPS); inv1 = 1.0 / (cs1 + HC_EPS)
        inv2 = 1.0 / (cs2 + HC_EPS); inv3 = 1.0 / (cs3 + HC_EPS)
        c00 *= inv0; c10 *= inv0; c20 *= inv0; c30 *= inv0
        c01 *= inv1; c11 *= inv1; c21 *= inv1; c31 *= inv1
        c02 *= inv2; c12 *= inv2; c22 *= inv2; c32 *= inv2
        c03 *= inv3; c13 *= inv3; c23 *= inv3; c33 *= inv3

    # store comb matrix (row-major)
    cb = t_off * 16
    tl.store(comb_ptr + cb + 0,  c00, mask=t_mask); tl.store(comb_ptr + cb + 1,  c01, mask=t_mask)
    tl.store(comb_ptr + cb + 2,  c02, mask=t_mask); tl.store(comb_ptr + cb + 3,  c03, mask=t_mask)
    tl.store(comb_ptr + cb + 4,  c10, mask=t_mask); tl.store(comb_ptr + cb + 5,  c11, mask=t_mask)
    tl.store(comb_ptr + cb + 6,  c12, mask=t_mask); tl.store(comb_ptr + cb + 7,  c13, mask=t_mask)
    tl.store(comb_ptr + cb + 8,  c20, mask=t_mask); tl.store(comb_ptr + cb + 9,  c21, mask=t_mask)
    tl.store(comb_ptr + cb + 10, c22, mask=t_mask); tl.store(comb_ptr + cb + 11, c23, mask=t_mask)
    tl.store(comb_ptr + cb + 12, c30, mask=t_mask); tl.store(comb_ptr + cb + 13, c31, mask=t_mask)
    tl.store(comb_ptr + cb + 14, c32, mask=t_mask); tl.store(comb_ptr + cb + 15, c33, mask=t_mask)

    # ---- fused y = sum_i(x_i * pre_i) ----
    xb = t_off * 4 * D
    for d_start in range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        x0 = tl.load(x_ptr + xb[:, None] + 0 * D + d_off[None, :],
                      mask=t_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + xb[:, None] + 1 * D + d_off[None, :],
                      mask=t_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + xb[:, None] + 2 * D + d_off[None, :],
                      mask=t_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + xb[:, None] + 3 * D + d_off[None, :],
                      mask=t_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        hin = (x0 * p0[:, None] + x1 * p1[:, None]
               + x2 * p2[:, None] + x3 * p3[:, None])
        tl.store(y_ptr + t_off[:, None] * D + d_off[None, :],
                 hin.to(tl.bfloat16),
                 mask=t_mask[:, None] & d_mask[None, :])


# ---------------------------------------------------------------------------
# Generic fused kernel (arbitrary hc_mult)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["hidden_size", "hc_hidden_size", "HC"],
)
@triton.jit
def _mhc_pre_generic_kernel(
    gemm_out_ptr,      # (T, hc_mult3), float32
    hc_scale_ptr,      # (3,),          float32
    hc_base_ptr,       # (hc_mult3,),   float32
    residual_ptr,      # (T, HC, H),    bfloat16
    post_mix_ptr,      # (T, HC),       float32  [out]
    comb_mix_ptr,      # (T, HC*HC),    float32  [out]
    layer_input_ptr,   # (T, H),        bfloat16 [out]
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Generic mHC pre kernel. One token per program."""
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    go_base = pid_n * (2 * HC + HC * HC)

    # ---- compute sqrsum for RMS normalisation ----
    sqrsum = 0.0
    for h_start in range(0, hc_hidden_size, BLOCK_H):
        h_off = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_off < hc_hidden_size
        i_idx = h_off // hidden_size
        h_idx = h_off % hidden_size
        v = tl.load(
            residual_ptr + pid_n * res_stride_n + i_idx * res_stride_i + h_idx * res_stride_h,
            mask=h_mask, other=0.0,
        ).to(tl.float32)
        sqrsum += tl.sum(v * v, axis=0)

    rms_inv = tl.math.rsqrt(sqrsum / hc_hidden_size + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    # ---- post_mix ----
    post_base = pid_n * HC
    for i in tl.static_range(HC):
        raw = tl.load(gemm_out_ptr + go_base + HC + i) * rms_inv * scale_1
        v = tl.sigmoid(raw + tl.load(hc_base_ptr + HC + i)) * hc_post_mult_value
        tl.store(post_mix_ptr + post_base + i, v)

    # ---- comb_mix: softmax-row + sinkhorn ----
    comb_base = pid_n * HC * HC
    for i in tl.static_range(HC):
        row_max = tl.load(gemm_out_ptr + go_base + 2 * HC + i * HC + 0) * rms_inv * scale_2 \
                  + tl.load(hc_base_ptr + 2 * HC + i * HC + 0)
        for j in tl.static_range(1, HC):
            val = tl.load(gemm_out_ptr + go_base + 2 * HC + i * HC + j) * rms_inv * scale_2 \
                  + tl.load(hc_base_ptr + 2 * HC + i * HC + j)
            row_max = tl.maximum(row_max, val)
        row_sum = 0.0
        for j in tl.static_range(HC):
            val = tl.load(gemm_out_ptr + go_base + 2 * HC + i * HC + j) * rms_inv * scale_2 \
                  + tl.load(hc_base_ptr + 2 * HC + i * HC + j)
            e = tl.exp(val - row_max)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, e)
            row_sum += e
        inv_rs = 1.0 / row_sum
        for j in tl.static_range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_rs + hc_sinkhorn_eps)

    for j in tl.static_range(HC):
        col_sum = 0.0
        for i in tl.static_range(HC):
            col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
        inv_cs = 1.0 / (col_sum + hc_sinkhorn_eps)
        for i in tl.static_range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_cs)

    for _ in tl.static_range(sinkhorn_repeat - 1):
        for i in tl.static_range(HC):
            row_sum = 0.0
            for j in tl.static_range(HC):
                row_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_rs = 1.0 / (row_sum + hc_sinkhorn_eps)
            for j in tl.static_range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_rs)
        for j in tl.static_range(HC):
            col_sum = 0.0
            for i in tl.static_range(HC):
                col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_cs = 1.0 / (col_sum + hc_sinkhorn_eps)
            for i in tl.static_range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_cs)

    # ---- layer_input = sum_i(pre_i * residual_i) ----
    res_base = pid_n * res_stride_n
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(HC):
            pre_k = (
                tl.sigmoid(
                    tl.load(gemm_out_ptr + go_base + k) * rms_inv * scale_0
                    + tl.load(hc_base_ptr + k)
                )
                + hc_pre_eps
            )
            rk = tl.load(
                residual_ptr + res_base + k * res_stride_i + h_offsets * res_stride_h,
                mask=h_mask, other=0.0,
            ).to(tl.float32)
            acc += pre_k * rk
        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ascend-optimised mHC Pre operator.

    Args:
        residual: (*, hc_mult, hidden_size), bfloat16
        fn: (hc_mult3, hc_mult*hidden_size), float32  where hc_mult3 = 2*hc_mult + hc_mult^2
        hc_scale: (3,), float32
        hc_base: (hc_mult3,), float32
        rms_eps: float
        hc_pre_eps: float
        hc_sinkhorn_eps: float
        hc_post_mult_value: float (expected to be 2.0)
        sinkhorn_repeat: int
        n_splits: int (unused, kept for API compatibility)

    Returns:
        post_mix:    (*, hc_mult, 1),        float32
        comb_mix:    (*, hc_mult, hc_mult),  float32
        layer_input: (*, hidden_size),       bfloat16
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    if hc_post_mult_value != 2.0:
        logger.warning(
            "mhc_pre: hc_post_mult_value=%s != 2.0; falling back to reference",
            hc_post_mult_value,
        )
        return mhc_pre_ref(
            residual, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat,
        )

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    assert fn.shape == (hc_mult3, hc_hidden_size)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]
    device = residual.device

    # bucket for autotune key stability
    if num_tokens <= 512:
        num_tokens_bucket = 1
    elif num_tokens <= 1024:
        num_tokens_bucket = 2
    elif num_tokens <= 2048:
        num_tokens_bucket = 3
    elif num_tokens <= 4096:
        num_tokens_bucket = 4
    else:
        num_tokens_bucket = 5

    # Step 1: GEMM (torch.mm, matches production MMAD on Ascend)
    x_flat = residual_flat.reshape(num_tokens, hc_hidden_size)
    fn_bf16 = _get_fn_bf16_cached(fn)
    gemm_out = torch.mm(x_flat, fn_bf16.t()).float()

    # Step 2: fused RMS-norm + sigmoid heads + Sinkhorn + weighted sum
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    if hc_mult == 4:
        # Fast path: vectorised BLOCK_T kernel keeps all 16 comb scalars in registers
        inv_rms = torch.empty(num_tokens, dtype=torch.float32, device=device)
        _rms_only_kernel[(num_tokens,)](
            x_flat, inv_rms,
            HC_D=hc_hidden_size,
            D_INV=1.0 / hc_hidden_size,
            NORM_EPS=rms_eps,
        )
        # scale mixes by inv_rms (in-place; gemm_out is a fresh allocation)
        gemm_out.mul_(inv_rms.unsqueeze(1))

        BLOCK_T = 32
        BLOCK_D = 256
        grid = (triton.cdiv(num_tokens, BLOCK_T),)
        _fused_heads_sinkhorn_yscale_hc4[grid](
            gemm_out, inv_rms, hc_scale, hc_base,
            residual_flat,
            post_mix, comb_mix, layer_input,
            T=num_tokens, D=hidden_size,
            HC_EPS=hc_sinkhorn_eps,
            SINKHORN_ITERS=sinkhorn_repeat,
            BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=1,
        )
    else:
        # Generic path: one token per program
        _mhc_pre_generic_kernel[(num_tokens,)](
            gemm_out, hc_scale, hc_base,
            residual_flat,
            post_mix, comb_mix, layer_input,
            num_tokens, num_tokens_bucket,
            residual_flat.stride(0),
            residual_flat.stride(1),
            residual_flat.stride(2),
            layer_input.stride(0),
            layer_input.stride(1),
            hidden_size, hc_hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            HC=hc_mult,
        )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference."""
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )
    hc_scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_expanded + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = _sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input
