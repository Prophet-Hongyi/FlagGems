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
Ascend implementation of mHC Post operator.

Computes:
    out[n, i, h] = post_layer_mix[n, i] * x[n, h]
                 + sum_j(comb_res_mix[n, j, i] * residual[n, j, h])

Public API matches flag_gems.fused.mhc.mhc_post.

Ascend optimizations:
- num_stages=1 configs only: NPU does not benefit from multi-stage software
  pipelining; extra stages waste register space.
- HC=4 fast path: all 20 scalars loaded into registers once per h-tile,
  enabling 4 accumulators computed with full ILP.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HC=4 fast-path kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["H"],
)
@triton.jit
def _mhc_post_kernel_hc4(
    a_ptr,    # comb_res_mix : (N, 4, 4), float32  — a[n, j, i]
    b_ptr,    # residual     : (N, 4, H), bfloat16
    c_ptr,    # post_layer_mix: (N, 4),   float32
    d_ptr,    # x            : (N, H),    bfloat16
    out_ptr,  # output       : (N, 4, H), bfloat16
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Grid: (N, cdiv(H, BLOCK_H)). One program per token × h-tile × all 4 streams."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base   = pid_n * 16
    c_base   = pid_n * 4
    b_base   = pid_n * 4 * H
    d_base   = pid_n * H
    out_base = pid_n * 4 * H

    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base + 0).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 4).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 5).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 6).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 7).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 8).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 9).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 10).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 11).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 12).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 13).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 14).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 15).to(tl.float32)

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)

    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

    tl.store(out_ptr + out_base + 0 * H + h_off, acc0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, acc1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, acc2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, acc3.to(tl.bfloat16), mask=h_mask)


# ---------------------------------------------------------------------------
# Generic kernel (arbitrary HC)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
    ],
    key=["H", "HC"],
)
@triton.jit
def _mhc_post_kernel_generic(
    a_ptr,    # comb_res_mix : (N, HC, HC), float32
    b_ptr,    # residual     : (N, HC, H),  bfloat16
    c_ptr,    # post_layer_mix: (N, HC),    float32
    d_ptr,    # x            : (N, H),      bfloat16
    out_ptr,  # output       : (N, HC, H),  bfloat16
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Grid: (N, HC, cdiv(H, BLOCK_H)). One program per token × stream i × h-tile."""
    pid_n = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base   = pid_n * HC * HC
    b_base   = pid_n * HC * H
    c_base   = pid_n * HC
    d_base   = pid_n * H
    out_base = pid_n * HC * H + pid_i * H

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    c_i    = tl.load(c_ptr + c_base + pid_i).to(tl.float32)

    acc = c_i * d_vals
    for j in tl.static_range(0, HC):
        a_ji = tl.load(a_ptr + a_base + j * HC + pid_i).to(tl.float32)
        b_j  = tl.load(b_ptr + b_base + j * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        acc += a_ji * b_j

    tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """
    mHC post-processing operator (Ascend implementation).

    Args:
        x: (N, H), bfloat16 — layer output
        residual: (N, hc_mult, H), bfloat16 — multi-head residual
        post_layer_mix: (N, hc_mult, 1) or (N, hc_mult), float32 — per-stream scale for x
        comb_res_mix: (N, hc_mult, hc_mult), float32 — combination matrix

    Returns:
        out: (N, hc_mult, H), bfloat16
    """
    logger.debug(
        "ASCEND MHC_POST FORWARD, x=%s, residual=%s, post_layer_mix=%s, comb_res_mix=%s",
        x.shape,
        residual.shape,
        post_layer_mix.shape,
        comb_res_mix.shape,
    )

    N, hc, H = residual.shape
    assert x.shape == (N, H)
    assert post_layer_mix.shape in ((N, hc, 1), (N, hc))
    assert comb_res_mix.shape == (N, hc, hc)

    out = torch.empty_like(residual)

    c = post_layer_mix.squeeze(-1).contiguous()  # (N, hc)
    a = comb_res_mix.contiguous()                # (N, hc, hc)
    b = residual.contiguous()                    # (N, hc, H)
    d = x.contiguous()                           # (N, H)

    if hc == 4:
        def _grid(META):
            return (N, triton.cdiv(H, META["BLOCK_H"]))

        _mhc_post_kernel_hc4[_grid](a, b, c, d, out, H=H)
    else:
        def _grid_generic(META):
            return (N, hc, triton.cdiv(H, META["BLOCK_H"]))

        _mhc_post_kernel_generic[_grid_generic](a, b, c, d, out, H=H, HC=hc)

    return out


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation."""
    y = (
        x.unsqueeze(-2) * post_layer_mix
        + torch.bmm(comb_res_mix.mT, residual.float())
    )
    return y.type_as(x)
