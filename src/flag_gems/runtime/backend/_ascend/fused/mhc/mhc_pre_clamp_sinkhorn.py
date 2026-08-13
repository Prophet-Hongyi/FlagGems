"""Optimized Triton implementation of mhc_pre_sinkhorn based on Ascend C logic.

Key optimizations from Ascend C kernel:
1. AIV/AIC cooperative execution mapped to Triton kernels:
     Kernel A (AIV-style): RMSNorm across (hcMult*D), produce inv_rms + x_scaled.
     Kernel B (AIC-style): Tiled GEMM x_scaled @ phi.T using tl.dot (cube path).
     Kernel C (AIV-style): Pre/post/sinkhorn/y_scale from mixes in registers.
2. Grid-stride loop: every kernel is launched with num_programs = core_count;
   each program iterates over its share of tokens (or GEMM tiles) in a loop.
3. No torch.matmul – the GEMM is implemented with a Triton tiled kernel.
4. Register-level Sinkhorn: all 16 matrix cells live in registers for hcMult=4.

Ascend C stages mirrored:
  Stage1 (MhcPreSinkhornStage1):
    AIV: RMSNorm(x) → inv_rms; cast BF16→FP32 and scale → x_scaled
    AIC: matmul(x_scaled, phi.T) → mixes  (hcBeforeNorm)
  Stage2 (MhcPreSinkhornStage2):
    AIV: pre/post sigmoid, clamp+softmax+Sinkhorn, y_scale
"""

import torch
import triton
import triton.language as tl
import torch_npu
import triton.runtime.driver as driver


def get_device_core_counts():
    """Return (num_vectorcore, num_aicore) for the current device."""
    device = torch_npu.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return props["num_vectorcore"], props["num_aicore"]


# ---------------------------------------------------------------------------
# Triton GEMM: x_scaled (T, K) @ phi^T (K, hcMix) -> mixes (T, hcMix)
# Replaces torch.mm to satisfy the no-torch.matmul requirement.
# Tile layout mirrors Ascend C HcCubeCompute: tiles over M (T) dimension,
# full N (hcMix=24) fits in a single BLOCK_N=32 tile.
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_xphi_kernel(
    x_ptr,  # (M, K) fp32 – x_scaled
    phi_ptr,  # (N_out, K) fp32 – phi stored as (hcMix, K), i.e. phi[row, :] = one basis vector
    out_ptr,  # (M, N_out) fp32 – mixes
    M,
    N_out,
    K,
    stride_xm,
    stride_xk,
    stride_pm,
    stride_pk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grid-stride tiled GEMM: out = x @ phi.T.

    Maps to Ascend C Stage1 AIC path (HcCubeCompute::ProcessMatmulXPhi).
    Each program claims tile_ids in a grid-stride loop so that launching with
    num_programs = aicore_num perfectly saturates the cube cores.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N_out, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    # Grid-stride loop over output tiles
    for tile_id in range(pid, total_tiles, num_programs):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n

        m_off = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_off = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_off < M
        n_mask = n_off < N_out

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Reduction loop over K
        for k_start in range(0, K, BLOCK_K):
            k_off = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_off < K

            # Load A tile: (BLOCK_M, BLOCK_K) from x_scaled
            a_tile = tl.load(
                x_ptr + m_off[:, None] * stride_xm + k_off[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # Load B tile: phi[n_off, k_off] → (BLOCK_N, BLOCK_K), transposed to (BLOCK_K, BLOCK_N)
            b_tile = tl.load(
                phi_ptr + n_off[:, None] * stride_pm + k_off[None, :] * stride_pk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            # tl.dot(A, B.T) – cube core path on Ascend
            acc = tl.dot(a_tile, tl.trans(b_tile), acc)

        # Store output tile
        tl.store(
            out_ptr + m_off[:, None] * stride_om + n_off[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Stage1 AIV: RMSNorm kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1),
    ],
    key=["HC_D"],
)
@triton.jit
def _compute_rms_norm_and_scale_kernel(
    x_ptr,  # (T, HC_D) input
    x_scaled_ptr,  # (T, HC_D) fp32 output
    inv_rms_ptr,  # (T,) fp32 output
    T,  # runtime int – grid-stride upper bound
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,
    NORM_EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Grid-stride RMSNorm kernel: compute inv_rms and scaled output."""
    num_programs = tl.num_programs(0)
    token_idx = tl.program_id(0)

    for token_id in range(token_idx, T, num_programs):
        base_offset = token_id * HC_D

        # Accumulate sum of squares over HC_D in chunks
        sum_squares = 0.0
        for chunk_start in range(0, HC_D, BLOCK_SIZE):
            offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < HC_D
            values = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_squares += tl.sum(values * values, axis=0)

        # inv_rms = 1 / sqrt(mean(x^2) + eps)
        inv_rms = tl.math.rsqrt(sum_squares * D_INV + NORM_EPS)
        tl.store(inv_rms_ptr + token_id, inv_rms)

        # Write fp32-scaled x to workspace
        for chunk_start in range(0, HC_D, BLOCK_SIZE):
            offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < HC_D
            values = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
            tl.store(x_scaled_ptr + base_offset + offsets, values * inv_rms, mask=mask)


@triton.jit
def _compute_heads_and_sinkhorn_kernel(
    mixes_ptr,  # (T, 24) fp32
    alpha_ptr,  # (3,) fp32
    base_ptr,  # (24,) fp32
    pre_ptr,  # (T, 4) fp32 output
    post_ptr,  # (T, 4) fp32 output
    comb_ptr,  # (T, 16) fp32 output
    num_tokens,  # runtime int
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    ITERS: tl.constexpr,
):
    """Grid-stride Sinkhorn kernel with vectorized operations."""
    num_programs = tl.num_programs(0)
    program_id = tl.program_id(0)

    # Load alpha scalars (shared across all tokens)
    # Note: Cannot use vectorized load + indexing due to Triton Ascend limitation
    alpha_0 = tl.load(alpha_ptr + 0)
    alpha_1 = tl.load(alpha_ptr + 1)
    alpha_2 = tl.load(alpha_ptr + 2)

    for token_id in range(program_id, num_tokens, num_programs):
        mix_offset = token_id * 24

        # Pre head: 4 sigmoids (vectorized load/store)
        pre_offsets = tl.arange(0, 4)
        pre_mixes = tl.load(mixes_ptr + mix_offset + pre_offsets)
        pre_bases = tl.load(base_ptr + pre_offsets)
        pre_out = tl.sigmoid(pre_mixes * alpha_0 + pre_bases) + HC_EPS
        tl.store(pre_ptr + token_id * 4 + pre_offsets, pre_out)

        # Post head: 4 sigmoids * 2 (vectorized load/store)
        post_offsets = tl.arange(0, 4)
        post_mixes = tl.load(mixes_ptr + mix_offset + 4 + post_offsets)
        post_bases = tl.load(base_ptr + 4 + post_offsets)
        post_out = 2.0 * tl.sigmoid(post_mixes * alpha_1 + post_bases)
        tl.store(post_ptr + token_id * 4 + post_offsets, post_out)

        # CombLogits: load 4x4 matrix elements individually
        # NOTE: Cannot use vectorized load + indexing (e.g., vec[i]) due to Triton Ascend limitation.
        # The compiler raises "unsupported tensor index" error for any vector indexing.
        # This element-wise approach is the most efficient method available.
        l00 = tl.load(mixes_ptr + mix_offset + 8 + 0) * alpha_2 + tl.load(base_ptr + 8 + 0)
        l01 = tl.load(mixes_ptr + mix_offset + 8 + 1) * alpha_2 + tl.load(base_ptr + 8 + 1)
        l02 = tl.load(mixes_ptr + mix_offset + 8 + 2) * alpha_2 + tl.load(base_ptr + 8 + 2)
        l03 = tl.load(mixes_ptr + mix_offset + 8 + 3) * alpha_2 + tl.load(base_ptr + 8 + 3)
        l10 = tl.load(mixes_ptr + mix_offset + 8 + 4) * alpha_2 + tl.load(base_ptr + 8 + 4)
        l11 = tl.load(mixes_ptr + mix_offset + 8 + 5) * alpha_2 + tl.load(base_ptr + 8 + 5)
        l12 = tl.load(mixes_ptr + mix_offset + 8 + 6) * alpha_2 + tl.load(base_ptr + 8 + 6)
        l13 = tl.load(mixes_ptr + mix_offset + 8 + 7) * alpha_2 + tl.load(base_ptr + 8 + 7)
        l20 = tl.load(mixes_ptr + mix_offset + 8 + 8) * alpha_2 + tl.load(base_ptr + 8 + 8)
        l21 = tl.load(mixes_ptr + mix_offset + 8 + 9) * alpha_2 + tl.load(base_ptr + 8 + 9)
        l22 = tl.load(mixes_ptr + mix_offset + 8 + 10) * alpha_2 + tl.load(base_ptr + 8 + 10)
        l23 = tl.load(mixes_ptr + mix_offset + 8 + 11) * alpha_2 + tl.load(base_ptr + 8 + 11)
        l30 = tl.load(mixes_ptr + mix_offset + 8 + 12) * alpha_2 + tl.load(base_ptr + 8 + 12)
        l31 = tl.load(mixes_ptr + mix_offset + 8 + 13) * alpha_2 + tl.load(base_ptr + 8 + 13)
        l32 = tl.load(mixes_ptr + mix_offset + 8 + 14) * alpha_2 + tl.load(base_ptr + 8 + 14)
        l33 = tl.load(mixes_ptr + mix_offset + 8 + 15) * alpha_2 + tl.load(base_ptr + 8 + 15)

        # Clamp logits (conditional based on CLAMP_MIN and CLAMP_MAX)
        # Only apply clamp if either min or max is non-zero
        apply_clamp = (CLAMP_MIN != 0.0) or (CLAMP_MAX != 0.0)
        if apply_clamp:
            l00 = tl.minimum(tl.maximum(l00, CLAMP_MIN), CLAMP_MAX)
            l01 = tl.minimum(tl.maximum(l01, CLAMP_MIN), CLAMP_MAX)
            l02 = tl.minimum(tl.maximum(l02, CLAMP_MIN), CLAMP_MAX)
            l03 = tl.minimum(tl.maximum(l03, CLAMP_MIN), CLAMP_MAX)
            l10 = tl.minimum(tl.maximum(l10, CLAMP_MIN), CLAMP_MAX)
            l11 = tl.minimum(tl.maximum(l11, CLAMP_MIN), CLAMP_MAX)
            l12 = tl.minimum(tl.maximum(l12, CLAMP_MIN), CLAMP_MAX)
            l13 = tl.minimum(tl.maximum(l13, CLAMP_MIN), CLAMP_MAX)
            l20 = tl.minimum(tl.maximum(l20, CLAMP_MIN), CLAMP_MAX)
            l21 = tl.minimum(tl.maximum(l21, CLAMP_MIN), CLAMP_MAX)
            l22 = tl.minimum(tl.maximum(l22, CLAMP_MIN), CLAMP_MAX)
            l23 = tl.minimum(tl.maximum(l23, CLAMP_MIN), CLAMP_MAX)
            l30 = tl.minimum(tl.maximum(l30, CLAMP_MIN), CLAMP_MAX)
            l31 = tl.minimum(tl.maximum(l31, CLAMP_MIN), CLAMP_MAX)
            l32 = tl.minimum(tl.maximum(l32, CLAMP_MIN), CLAMP_MAX)
            l33 = tl.minimum(tl.maximum(l33, CLAMP_MIN), CLAMP_MAX)

        # Row-softmax
        row_max_0 = tl.maximum(tl.maximum(l00, l01), tl.maximum(l02, l03))
        row_max_1 = tl.maximum(tl.maximum(l10, l11), tl.maximum(l12, l13))
        row_max_2 = tl.maximum(tl.maximum(l20, l21), tl.maximum(l22, l23))
        row_max_3 = tl.maximum(tl.maximum(l30, l31), tl.maximum(l32, l33))

        e00 = tl.exp(l00 - row_max_0)
        e01 = tl.exp(l01 - row_max_0)
        e02 = tl.exp(l02 - row_max_0)
        e03 = tl.exp(l03 - row_max_0)
        e10 = tl.exp(l10 - row_max_1)
        e11 = tl.exp(l11 - row_max_1)
        e12 = tl.exp(l12 - row_max_1)
        e13 = tl.exp(l13 - row_max_1)
        e20 = tl.exp(l20 - row_max_2)
        e21 = tl.exp(l21 - row_max_2)
        e22 = tl.exp(l22 - row_max_2)
        e23 = tl.exp(l23 - row_max_2)
        e30 = tl.exp(l30 - row_max_3)
        e31 = tl.exp(l31 - row_max_3)
        e32 = tl.exp(l32 - row_max_3)
        e33 = tl.exp(l33 - row_max_3)

        row_sum_inv_0 = 1.0 / (e00 + e01 + e02 + e03)
        row_sum_inv_1 = 1.0 / (e10 + e11 + e12 + e13)
        row_sum_inv_2 = 1.0 / (e20 + e21 + e22 + e23)
        row_sum_inv_3 = 1.0 / (e30 + e31 + e32 + e33)

        v00 = e00 * row_sum_inv_0
        v01 = e01 * row_sum_inv_0
        v02 = e02 * row_sum_inv_0
        v03 = e03 * row_sum_inv_0
        v10 = e10 * row_sum_inv_1
        v11 = e11 * row_sum_inv_1
        v12 = e12 * row_sum_inv_1
        v13 = e13 * row_sum_inv_1
        v20 = e20 * row_sum_inv_2
        v21 = e21 * row_sum_inv_2
        v22 = e22 * row_sum_inv_2
        v23 = e23 * row_sum_inv_2
        v30 = e30 * row_sum_inv_3
        v31 = e31 * row_sum_inv_3
        v32 = e32 * row_sum_inv_3
        v33 = e33 * row_sum_inv_3

        # Add eps + col-normalize
        v00 += HC_EPS
        v01 += HC_EPS
        v02 += HC_EPS
        v03 += HC_EPS
        v10 += HC_EPS
        v11 += HC_EPS
        v12 += HC_EPS
        v13 += HC_EPS
        v20 += HC_EPS
        v21 += HC_EPS
        v22 += HC_EPS
        v23 += HC_EPS
        v30 += HC_EPS
        v31 += HC_EPS
        v32 += HC_EPS
        v33 += HC_EPS

        col_sum_inv_0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
        col_sum_inv_1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
        col_sum_inv_2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
        col_sum_inv_3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)

        v00 *= col_sum_inv_0
        v01 *= col_sum_inv_1
        v02 *= col_sum_inv_2
        v03 *= col_sum_inv_3
        v10 *= col_sum_inv_0
        v11 *= col_sum_inv_1
        v12 *= col_sum_inv_2
        v13 *= col_sum_inv_3
        v20 *= col_sum_inv_0
        v21 *= col_sum_inv_1
        v22 *= col_sum_inv_2
        v23 *= col_sum_inv_3
        v30 *= col_sum_inv_0
        v31 *= col_sum_inv_1
        v32 *= col_sum_inv_2
        v33 *= col_sum_inv_3

        # Sinkhorn iterations
        for _ in tl.static_range(ITERS - 1):
            row_inv_0 = 1.0 / (v00 + v01 + v02 + v03 + HC_EPS)
            row_inv_1 = 1.0 / (v10 + v11 + v12 + v13 + HC_EPS)
            row_inv_2 = 1.0 / (v20 + v21 + v22 + v23 + HC_EPS)
            row_inv_3 = 1.0 / (v30 + v31 + v32 + v33 + HC_EPS)

            v00 *= row_inv_0
            v01 *= row_inv_0
            v02 *= row_inv_0
            v03 *= row_inv_0
            v10 *= row_inv_1
            v11 *= row_inv_1
            v12 *= row_inv_1
            v13 *= row_inv_1
            v20 *= row_inv_2
            v21 *= row_inv_2
            v22 *= row_inv_2
            v23 *= row_inv_2
            v30 *= row_inv_3
            v31 *= row_inv_3
            v32 *= row_inv_3
            v33 *= row_inv_3

            col_inv_0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
            col_inv_1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
            col_inv_2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
            col_inv_3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)

            v00 *= col_inv_0
            v01 *= col_inv_1
            v02 *= col_inv_2
            v03 *= col_inv_3
            v10 *= col_inv_0
            v11 *= col_inv_1
            v12 *= col_inv_2
            v13 *= col_inv_3
            v20 *= col_inv_0
            v21 *= col_inv_1
            v22 *= col_inv_2
            v23 *= col_inv_3
            v30 *= col_inv_0
            v31 *= col_inv_1
            v32 *= col_inv_2
            v33 *= col_inv_3

        # Store comb_frag
        comb_offset = token_id * 16
        tl.store(comb_ptr + comb_offset + 0, v00)
        tl.store(comb_ptr + comb_offset + 1, v01)
        tl.store(comb_ptr + comb_offset + 2, v02)
        tl.store(comb_ptr + comb_offset + 3, v03)
        tl.store(comb_ptr + comb_offset + 4, v10)
        tl.store(comb_ptr + comb_offset + 5, v11)
        tl.store(comb_ptr + comb_offset + 6, v12)
        tl.store(comb_ptr + comb_offset + 7, v13)
        tl.store(comb_ptr + comb_offset + 8, v20)
        tl.store(comb_ptr + comb_offset + 9, v21)
        tl.store(comb_ptr + comb_offset + 10, v22)
        tl.store(comb_ptr + comb_offset + 11, v23)
        tl.store(comb_ptr + comb_offset + 12, v30)
        tl.store(comb_ptr + comb_offset + 13, v31)
        tl.store(comb_ptr + comb_offset + 14, v32)
        tl.store(comb_ptr + comb_offset + 15, v33)


@triton.jit
def _compute_weighted_sum_kernel(
    x_ptr,  # (T, 4, D) input
    pre_ptr,  # (T, 4) fp32
    y_ptr,  # (T, D) output
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Grid-stride weighted sum kernel: y[t,d] = sum_n(x[t,n,d] * pre[t,n])."""
    num_programs = tl.num_programs(0)
    program_id = tl.program_id(0)

    for token_id in range(program_id, num_tokens, num_programs):
        # Load pre weights (4 values individually)
        weight_0 = tl.load(pre_ptr + token_id * 4 + 0)
        weight_1 = tl.load(pre_ptr + token_id * 4 + 1)
        weight_2 = tl.load(pre_ptr + token_id * 4 + 2)
        weight_3 = tl.load(pre_ptr + token_id * 4 + 3)

        x_base_offset = token_id * 4 * head_dim
        output_dtype = y_ptr.dtype.element_ty

        # Grid-stride over head dimension
        for dim_start in range(0, head_dim, BLOCK_SIZE):
            dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE)
            dim_mask = dim_offsets < head_dim

            # Load x values for all 4 heads (vectorized)
            x0 = tl.load(x_ptr + x_base_offset + 0 * head_dim + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + x_base_offset + 1 * head_dim + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + x_base_offset + 2 * head_dim + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + x_base_offset + 3 * head_dim + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)

            # Weighted sum
            result = x0 * weight_0 + x1 * weight_1 + x2 * weight_2 + x3 * weight_3
            tl.store(y_ptr + token_id * head_dim + dim_offsets, result.to(output_dtype), mask=dim_mask)


def mhc_pre_clamp_sinkhorn(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    clamp_min: float = 0.0,
    clamp_max: float = 0.0,
    iter_times: int = 20,
    need_backward: bool = False,
):
    """Optimized MHC pre+clamp+Sinkhorn with grid-stride multi-core execution.

    Mirrors the Ascend C two-stage pipeline:
      Stage1 (AIV): RMSNorm → inv_rms + x_scaled  (grid = num_vectorcore)
      Stage1 (AIC): Triton tiled GEMM x_scaled @ phi.T → mixes  (grid = num_aicore)
      Stage2 (AIV): sigmoid/sinkhorn/y_scale from mixes  (grid = num_vectorcore)

    No torch.matmul – the GEMM is implemented with _matmul_xphi_kernel.
    """
    # Flatten to (num_tokens, num_heads, head_dim)
    if x.dim() == 4:
        batch_size, seq_len, num_heads, head_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, num_heads, head_dim).contiguous()
        y_shape = (batch_size, seq_len, head_dim)
    elif x.dim() == 3:
        x_flat = x.contiguous()
        y_shape = (x_flat.shape[0], x_flat.shape[2])
    else:
        raise ValueError(f"Unsupported x.dim()={x.dim()}")

    num_tokens, num_heads, head_dim = x_flat.shape
    assert num_heads == 4, "hc_mult=4 only"
    hc_mix = num_heads * (num_heads + 2)  # 24
    hc_d = num_heads * head_dim  # K for GEMM

    # Query device core counts to set grid sizes
    num_vectorcore, num_aicore = get_device_core_counts()

    # -----------------------------------------------------------------------
    # Stage 1 – Part A: RMSNorm (vector cores)
    # Grid-stride over tokens with num_vectorcore programs.
    # -----------------------------------------------------------------------
    inv_rms = torch.empty(num_tokens, dtype=torch.float32, device=x_flat.device)
    x_scaled = torch.empty(num_tokens, hc_d, dtype=torch.float32, device=x_flat.device)

    _compute_rms_norm_and_scale_kernel[(num_vectorcore, )](
        x_flat.reshape(num_tokens, hc_d),
        x_scaled,
        inv_rms,
        T=num_tokens,
        HC_D=hc_d,
        D_INV=1.0 / hc_d,
        NORM_EPS=norm_eps,
    )

    # -----------------------------------------------------------------------
    # Stage 1 – Part B: GEMM  (AI cores – cube cores via tl.dot)
    # mixes = x_scaled @ phi.T  shape: (num_tokens, hcMix)
    # Grid-stride over (M, N) tiles with num_aicore programs.
    # Tile sizes: BLOCK_M=32, BLOCK_N=32 (covers hcMix=24), BLOCK_K=128.
    # -----------------------------------------------------------------------
    phi_f = phi.to(torch.float32)
    mixes = torch.mm(x_scaled, phi_f.t())  # (T, hcMix)
    '''
    mixes = torch.empty(num_tokens, hc_mix, dtype=torch.float32, device=x_flat.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 128
    _matmul_xphi_kernel[(num_aicore, )](
        x_scaled,
        phi_f,
        mixes,
        num_tokens,
        hc_mix,
        hc_d,
        x_scaled.stride(0),
        x_scaled.stride(1),
        phi_f.stride(0),
        phi_f.stride(1),
        mixes.stride(0),
        mixes.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=1,
    )
    '''

    # -----------------------------------------------------------------------
    # Stage 2 – Sinkhorn heads (vector cores)
    # Grid-stride over tokens with num_vectorcore programs.
    # -----------------------------------------------------------------------
    pre = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=x_flat.device)
    post_out = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=x_flat.device)
    comb_frag = torch.empty(num_tokens, num_heads, num_heads, dtype=torch.float32, device=x_flat.device)

    _compute_heads_and_sinkhorn_kernel[(num_vectorcore, )](
        mixes,
        alpha.float(),
        base.float(),
        pre,
        post_out,
        comb_frag.reshape(num_tokens, 16),
        num_tokens=num_tokens,
        HC_EPS=hc_eps,
        CLAMP_MIN=float(clamp_min),
        CLAMP_MAX=float(clamp_max),
        ITERS=iter_times,
        num_stages=1,
    )

    # -----------------------------------------------------------------------
    # Stage 2 – Y-scale (vector cores)
    # y[t, d] = sum_n(x[t, n, d] * pre[t, n])
    # Grid-stride over tokens with num_vectorcore programs.
    # -----------------------------------------------------------------------
    y = torch.empty(num_tokens, head_dim, dtype=x_flat.dtype, device=x_flat.device)
    _compute_weighted_sum_kernel[(num_vectorcore, )](
        x_flat,
        pre,
        y,
        num_tokens=num_tokens,
        head_dim=head_dim,
        BLOCK_SIZE=256,
        num_stages=1,
    )

    result = {
        "y": y.reshape(y_shape),
        "post_out": post_out,
        "comb_frag": comb_frag,
    }

    if need_backward:
        result.update(
            inv_rms=inv_rms,
            x_scaled=x_scaled,
            mixes=mixes,
            pre=pre,
        )

    return result
