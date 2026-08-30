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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

ROUTER_SPLITK_BLOCK_M = 16
ROUTER_SPLITK_BLOCK_N = 64
ROUTER_SPLITK_BLOCK_K = 64
ROUTER_SPLITK = 4
ROUTER_SPLITK_NUM_WARPS = 4
ROUTER_SPLITK_NUM_STAGES = 3


@libentry()
@triton.jit
def mm_kernel_router_splitk(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offset_am = pid_m * BLOCK_M
    offset_bn = pid_n * BLOCK_N
    offs_am = offset_am + tl.arange(0, BLOCK_M)
    offs_bn = offset_bn + tl.arange(0, BLOCK_N)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end):
        offset_k = k * BLOCK_K
        offs_k = offset_k + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    offs_cm = offset_am + tl.arange(0, BLOCK_M)
    offs_cn = offset_bn + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)


def router_splitk_scenario(x, weight, M, N, K):
    return (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and M < 2048
        and N < 2048
        and K >= 4096
        and K % (ROUTER_SPLITK_BLOCK_K * ROUTER_SPLITK) == 0
    )


def router_splitk_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS_THEAD ROUTER_GEMM_SPLITK, [shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    grid = (
        triton.cdiv(M, ROUTER_SPLITK_BLOCK_M)
        * triton.cdiv(N, ROUTER_SPLITK_BLOCK_N),
        ROUTER_SPLITK,
    )
    with torch_device_fn.device(a.device):
        mm_kernel_router_splitk[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=ROUTER_SPLITK_BLOCK_M,
            BLOCK_N=ROUTER_SPLITK_BLOCK_N,
            BLOCK_K=ROUTER_SPLITK_BLOCK_K,
            SPLIT_K=ROUTER_SPLITK,
            num_warps=ROUTER_SPLITK_NUM_WARPS,
            num_stages=ROUTER_SPLITK_NUM_STAGES,
        )
    return c


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """BF16 by BF16 router GEMM with FP32 output."""
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)
    b = weight.t()
    if router_splitk_scenario(x, weight, M, N, K):
        c.zero_()
        return router_splitk_mm(x, b, c, M, N, K)

    from flag_gems.ops.mm import general_mm

    return general_mm(x, b.contiguous(), c, M, N, K)
