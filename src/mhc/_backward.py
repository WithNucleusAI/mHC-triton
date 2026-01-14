"""Backward Triton kernels for Manifold-Constrained Hyper-Connections.

Key optimization: Recomputation during backward to avoid storing intermediate states.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# -----------------------------------------------------------------------------
# Sinkhorn Backward Kernel
# Recomputes forward pass to save memory (20x reduction)
# -----------------------------------------------------------------------------

@triton.jit
def _sinkhorn_backward_kernel(
    M_orig_ptr, dP_ptr, dM_ptr,
    batch,
    num_iters: tl.constexpr,
    eps: tl.constexpr,
    NUM_STREAMS: tl.constexpr,
):
    """Backward through Sinkhorn with O(T²) recomputation."""
    pid = tl.program_id(0)
    if pid >= batch:
        return

    base = pid * 16

    # Load original input and compute sign for abs() backward
    m00_orig = tl.load(M_orig_ptr + base + 0).to(tl.float32)
    m01_orig = tl.load(M_orig_ptr + base + 1).to(tl.float32)
    m02_orig = tl.load(M_orig_ptr + base + 2).to(tl.float32)
    m03_orig = tl.load(M_orig_ptr + base + 3).to(tl.float32)
    m10_orig = tl.load(M_orig_ptr + base + 4).to(tl.float32)
    m11_orig = tl.load(M_orig_ptr + base + 5).to(tl.float32)
    m12_orig = tl.load(M_orig_ptr + base + 6).to(tl.float32)
    m13_orig = tl.load(M_orig_ptr + base + 7).to(tl.float32)
    m20_orig = tl.load(M_orig_ptr + base + 8).to(tl.float32)
    m21_orig = tl.load(M_orig_ptr + base + 9).to(tl.float32)
    m22_orig = tl.load(M_orig_ptr + base + 10).to(tl.float32)
    m23_orig = tl.load(M_orig_ptr + base + 11).to(tl.float32)
    m30_orig = tl.load(M_orig_ptr + base + 12).to(tl.float32)
    m31_orig = tl.load(M_orig_ptr + base + 13).to(tl.float32)
    m32_orig = tl.load(M_orig_ptr + base + 14).to(tl.float32)
    m33_orig = tl.load(M_orig_ptr + base + 15).to(tl.float32)

    # Sign for abs() backward
    s00 = tl.where(m00_orig >= 0, 1.0, -1.0)
    s01 = tl.where(m01_orig >= 0, 1.0, -1.0)
    s02 = tl.where(m02_orig >= 0, 1.0, -1.0)
    s03 = tl.where(m03_orig >= 0, 1.0, -1.0)
    s10 = tl.where(m10_orig >= 0, 1.0, -1.0)
    s11 = tl.where(m11_orig >= 0, 1.0, -1.0)
    s12 = tl.where(m12_orig >= 0, 1.0, -1.0)
    s13 = tl.where(m13_orig >= 0, 1.0, -1.0)
    s20 = tl.where(m20_orig >= 0, 1.0, -1.0)
    s21 = tl.where(m21_orig >= 0, 1.0, -1.0)
    s22 = tl.where(m22_orig >= 0, 1.0, -1.0)
    s23 = tl.where(m23_orig >= 0, 1.0, -1.0)
    s30 = tl.where(m30_orig >= 0, 1.0, -1.0)
    s31 = tl.where(m31_orig >= 0, 1.0, -1.0)
    s32 = tl.where(m32_orig >= 0, 1.0, -1.0)
    s33 = tl.where(m33_orig >= 0, 1.0, -1.0)

    # Load upstream gradient
    d00 = tl.load(dP_ptr + base + 0).to(tl.float32)
    d01 = tl.load(dP_ptr + base + 1).to(tl.float32)
    d02 = tl.load(dP_ptr + base + 2).to(tl.float32)
    d03 = tl.load(dP_ptr + base + 3).to(tl.float32)
    d10 = tl.load(dP_ptr + base + 4).to(tl.float32)
    d11 = tl.load(dP_ptr + base + 5).to(tl.float32)
    d12 = tl.load(dP_ptr + base + 6).to(tl.float32)
    d13 = tl.load(dP_ptr + base + 7).to(tl.float32)
    d20 = tl.load(dP_ptr + base + 8).to(tl.float32)
    d21 = tl.load(dP_ptr + base + 9).to(tl.float32)
    d22 = tl.load(dP_ptr + base + 10).to(tl.float32)
    d23 = tl.load(dP_ptr + base + 11).to(tl.float32)
    d30 = tl.load(dP_ptr + base + 12).to(tl.float32)
    d31 = tl.load(dP_ptr + base + 13).to(tl.float32)
    d32 = tl.load(dP_ptr + base + 14).to(tl.float32)
    d33 = tl.load(dP_ptr + base + 15).to(tl.float32)

    # Backward through iterations (reverse order with recomputation)
    for _iter in range(num_iters):
        target_iter = num_iters - 1 - _iter

        # Recompute forward to target state
        m00 = tl.abs(m00_orig) + eps
        m01 = tl.abs(m01_orig) + eps
        m02 = tl.abs(m02_orig) + eps
        m03 = tl.abs(m03_orig) + eps
        m10 = tl.abs(m10_orig) + eps
        m11 = tl.abs(m11_orig) + eps
        m12 = tl.abs(m12_orig) + eps
        m13 = tl.abs(m13_orig) + eps
        m20 = tl.abs(m20_orig) + eps
        m21 = tl.abs(m21_orig) + eps
        m22 = tl.abs(m22_orig) + eps
        m23 = tl.abs(m23_orig) + eps
        m30 = tl.abs(m30_orig) + eps
        m31 = tl.abs(m31_orig) + eps
        m32 = tl.abs(m32_orig) + eps
        m33 = tl.abs(m33_orig) + eps

        for _fwd in range(num_iters):
            if _fwd < target_iter:
                r0 = m00 + m01 + m02 + m03 + eps
                r1 = m10 + m11 + m12 + m13 + eps
                r2 = m20 + m21 + m22 + m23 + eps
                r3 = m30 + m31 + m32 + m33 + eps
                m00 /= r0; m01 /= r0; m02 /= r0; m03 /= r0
                m10 /= r1; m11 /= r1; m12 /= r1; m13 /= r1
                m20 /= r2; m21 /= r2; m22 /= r2; m23 /= r2
                m30 /= r3; m31 /= r3; m32 /= r3; m33 /= r3

                c0 = m00 + m10 + m20 + m30 + eps
                c1 = m01 + m11 + m21 + m31 + eps
                c2 = m02 + m12 + m22 + m32 + eps
                c3 = m03 + m13 + m23 + m33 + eps
                m00 /= c0; m10 /= c0; m20 /= c0; m30 /= c0
                m01 /= c1; m11 /= c1; m21 /= c1; m31 /= c1
                m02 /= c2; m12 /= c2; m22 /= c2; m32 /= c2
                m03 /= c3; m13 /= c3; m23 /= c3; m33 /= c3

        # Forward step at target_iter
        r0 = m00 + m01 + m02 + m03 + eps
        r1 = m10 + m11 + m12 + m13 + eps
        r2 = m20 + m21 + m22 + m23 + eps
        r3 = m30 + m31 + m32 + m33 + eps

        mr00 = m00 / r0; mr01 = m01 / r0; mr02 = m02 / r0; mr03 = m03 / r0
        mr10 = m10 / r1; mr11 = m11 / r1; mr12 = m12 / r1; mr13 = m13 / r1
        mr20 = m20 / r2; mr21 = m21 / r2; mr22 = m22 / r2; mr23 = m23 / r2
        mr30 = m30 / r3; mr31 = m31 / r3; mr32 = m32 / r3; mr33 = m33 / r3

        c0 = mr00 + mr10 + mr20 + mr30 + eps
        c1 = mr01 + mr11 + mr21 + mr31 + eps
        c2 = mr02 + mr12 + mr22 + mr32 + eps
        c3 = mr03 + mr13 + mr23 + mr33 + eps

        mc00 = mr00 / c0; mc10 = mr10 / c0; mc20 = mr20 / c0; mc30 = mr30 / c0
        mc01 = mr01 / c1; mc11 = mr11 / c1; mc21 = mr21 / c1; mc31 = mr31 / c1
        mc02 = mr02 / c2; mc12 = mr12 / c2; mc22 = mr22 / c2; mc32 = mr32 / c2
        mc03 = mr03 / c3; mc13 = mr13 / c3; mc23 = mr23 / c3; mc33 = mr33 / c3

        # Backward through column norm: dx_i = (dy_i - <dy, y>) / s
        dot0 = d00 * mc00 + d10 * mc10 + d20 * mc20 + d30 * mc30
        dot1 = d01 * mc01 + d11 * mc11 + d21 * mc21 + d31 * mc31
        dot2 = d02 * mc02 + d12 * mc12 + d22 * mc22 + d32 * mc32
        dot3 = d03 * mc03 + d13 * mc13 + d23 * mc23 + d33 * mc33

        dr00 = (d00 - dot0) / c0; dr10 = (d10 - dot0) / c0
        dr20 = (d20 - dot0) / c0; dr30 = (d30 - dot0) / c0
        dr01 = (d01 - dot1) / c1; dr11 = (d11 - dot1) / c1
        dr21 = (d21 - dot1) / c1; dr31 = (d31 - dot1) / c1
        dr02 = (d02 - dot2) / c2; dr12 = (d12 - dot2) / c2
        dr22 = (d22 - dot2) / c2; dr32 = (d32 - dot2) / c2
        dr03 = (d03 - dot3) / c3; dr13 = (d13 - dot3) / c3
        dr23 = (d23 - dot3) / c3; dr33 = (d33 - dot3) / c3

        # Backward through row norm
        dot0 = dr00 * mr00 + dr01 * mr01 + dr02 * mr02 + dr03 * mr03
        dot1 = dr10 * mr10 + dr11 * mr11 + dr12 * mr12 + dr13 * mr13
        dot2 = dr20 * mr20 + dr21 * mr21 + dr22 * mr22 + dr23 * mr23
        dot3 = dr30 * mr30 + dr31 * mr31 + dr32 * mr32 + dr33 * mr33

        d00 = (dr00 - dot0) / r0; d01 = (dr01 - dot0) / r0
        d02 = (dr02 - dot0) / r0; d03 = (dr03 - dot0) / r0
        d10 = (dr10 - dot1) / r1; d11 = (dr11 - dot1) / r1
        d12 = (dr12 - dot1) / r1; d13 = (dr13 - dot1) / r1
        d20 = (dr20 - dot2) / r2; d21 = (dr21 - dot2) / r2
        d22 = (dr22 - dot2) / r2; d23 = (dr23 - dot2) / r2
        d30 = (dr30 - dot3) / r3; d31 = (dr31 - dot3) / r3
        d32 = (dr32 - dot3) / r3; d33 = (dr33 - dot3) / r3

    # Multiply by sign for abs()
    d00 *= s00; d01 *= s01; d02 *= s02; d03 *= s03
    d10 *= s10; d11 *= s11; d12 *= s12; d13 *= s13
    d20 *= s20; d21 *= s21; d22 *= s22; d23 *= s23
    d30 *= s30; d31 *= s31; d32 *= s32; d33 *= s33

    # Store
    tl.store(dM_ptr + base + 0, d00)
    tl.store(dM_ptr + base + 1, d01)
    tl.store(dM_ptr + base + 2, d02)
    tl.store(dM_ptr + base + 3, d03)
    tl.store(dM_ptr + base + 4, d10)
    tl.store(dM_ptr + base + 5, d11)
    tl.store(dM_ptr + base + 6, d12)
    tl.store(dM_ptr + base + 7, d13)
    tl.store(dM_ptr + base + 8, d20)
    tl.store(dM_ptr + base + 9, d21)
    tl.store(dM_ptr + base + 10, d22)
    tl.store(dM_ptr + base + 11, d23)
    tl.store(dM_ptr + base + 12, d30)
    tl.store(dM_ptr + base + 13, d31)
    tl.store(dM_ptr + base + 14, d32)
    tl.store(dM_ptr + base + 15, d33)


# -----------------------------------------------------------------------------
# Add Residual Backward Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _add_residual_backward_kernel(
    dH_new_ptr, branch_output_ptr, H_post_ptr,
    dH_residual_ptr, d_branch_output_ptr, dH_post_partial_ptr,
    batch, seq, dim,
    stride_h_b, stride_h_s, stride_h_n, stride_h_d,
    stride_bo_b, stride_bo_s, stride_bo_d,
    BLOCK_DIM: tl.constexpr,
):
    """Backward: H_new = H_residual + H_post * branch_output"""
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bs // seq
    s = pid_bs % seq

    d_offs = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d_offs < dim

    # Load H_post and branch_output
    hp_base = H_post_ptr + b * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)

    bo_base = branch_output_ptr + b * stride_bo_b + s * stride_bo_s
    branch = tl.load(bo_base + d_offs * stride_bo_d, mask=d_mask, other=0.0).to(tl.float32)

    # Load dH_new
    dh_base = dH_new_ptr + b * stride_h_b + s * stride_h_s
    dh0 = tl.load(dh_base + 0 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    dh1 = tl.load(dh_base + 1 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    dh2 = tl.load(dh_base + 2 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    dh3 = tl.load(dh_base + 3 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)

    # dH_residual = dH_new
    dhr_base = dH_residual_ptr + b * stride_h_b + s * stride_h_s
    tl.store(dhr_base + 0 * stride_h_n + d_offs * stride_h_d, dh0, mask=d_mask)
    tl.store(dhr_base + 1 * stride_h_n + d_offs * stride_h_d, dh1, mask=d_mask)
    tl.store(dhr_base + 2 * stride_h_n + d_offs * stride_h_d, dh2, mask=d_mask)
    tl.store(dhr_base + 3 * stride_h_n + d_offs * stride_h_d, dh3, mask=d_mask)

    # d_branch_output = sum_n(H_post[n] * dH_new[n])
    d_bo = hp0 * dh0 + hp1 * dh1 + hp2 * dh2 + hp3 * dh3
    d_bo_base = d_branch_output_ptr + b * stride_bo_b + s * stride_bo_s
    tl.store(d_bo_base + d_offs * stride_bo_d, d_bo, mask=d_mask)

    # Partial sums for dH_post
    num_d_blocks = tl.cdiv(dim, BLOCK_DIM)
    partial_base = dH_post_partial_ptr + (b * seq + s) * num_d_blocks * 4 + pid_d * 4
    tl.store(partial_base + 0, tl.sum(tl.where(d_mask, dh0 * branch, 0.0)))
    tl.store(partial_base + 1, tl.sum(tl.where(d_mask, dh1 * branch, 0.0)))
    tl.store(partial_base + 2, tl.sum(tl.where(d_mask, dh2 * branch, 0.0)))
    tl.store(partial_base + 3, tl.sum(tl.where(d_mask, dh3 * branch, 0.0)))


@triton.jit
def _reduce_dH_post_kernel(dH_post_partial_ptr, dH_post_ptr, batch, seq, num_d_blocks):
    """Reduce partial sums for dH_post."""
    b = tl.program_id(0)
    if b >= batch:
        return

    acc0, acc1, acc2, acc3 = 0.0, 0.0, 0.0, 0.0
    for s in range(seq):
        for d_blk in range(num_d_blocks):
            base = (b * seq + s) * num_d_blocks * 4 + d_blk * 4
            acc0 += tl.load(dH_post_partial_ptr + base + 0).to(tl.float32)
            acc1 += tl.load(dH_post_partial_ptr + base + 1).to(tl.float32)
            acc2 += tl.load(dH_post_partial_ptr + base + 2).to(tl.float32)
            acc3 += tl.load(dH_post_partial_ptr + base + 3).to(tl.float32)

    out_base = dH_post_ptr + b * 4
    tl.store(out_base + 0, acc0)
    tl.store(out_base + 1, acc1)
    tl.store(out_base + 2, acc2)
    tl.store(out_base + 3, acc3)


# -----------------------------------------------------------------------------
# Stream Mix Backward Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _stream_mix_backward_kernel(
    d_branch_input_ptr, dH_residual_ptr,
    H_ptr, H_pre_ptr, H_res_ptr,
    dH_ptr, dH_pre_partial_ptr, dH_res_partial_ptr,
    batch, seq, dim,
    stride_h_b, stride_h_s, stride_h_n, stride_h_d,
    stride_bi_b, stride_bi_s, stride_bi_d,
    BLOCK_DIM: tl.constexpr,
):
    """Backward through stream mixing."""
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bs // seq
    s = pid_bs % seq

    d_offs = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d_offs < dim

    # Load H[b,s,:,:]
    h_base = H_ptr + b * stride_h_b + s * stride_h_s
    h0 = tl.load(h_base + 0 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h1 = tl.load(h_base + 1 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h2 = tl.load(h_base + 2 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h3 = tl.load(h_base + 3 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)

    # Load weights
    hp_base = H_pre_ptr + b * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)

    hr_base = H_res_ptr + b * 16
    hr00 = tl.load(hr_base + 0).to(tl.float32)
    hr01 = tl.load(hr_base + 1).to(tl.float32)
    hr02 = tl.load(hr_base + 2).to(tl.float32)
    hr03 = tl.load(hr_base + 3).to(tl.float32)
    hr10 = tl.load(hr_base + 4).to(tl.float32)
    hr11 = tl.load(hr_base + 5).to(tl.float32)
    hr12 = tl.load(hr_base + 6).to(tl.float32)
    hr13 = tl.load(hr_base + 7).to(tl.float32)
    hr20 = tl.load(hr_base + 8).to(tl.float32)
    hr21 = tl.load(hr_base + 9).to(tl.float32)
    hr22 = tl.load(hr_base + 10).to(tl.float32)
    hr23 = tl.load(hr_base + 11).to(tl.float32)
    hr30 = tl.load(hr_base + 12).to(tl.float32)
    hr31 = tl.load(hr_base + 13).to(tl.float32)
    hr32 = tl.load(hr_base + 14).to(tl.float32)
    hr33 = tl.load(hr_base + 15).to(tl.float32)

    # Load gradients
    dbi_base = d_branch_input_ptr + b * stride_bi_b + s * stride_bi_s
    d_bi = tl.load(dbi_base + d_offs * stride_bi_d, mask=d_mask, other=0.0).to(tl.float32)

    dhr_base = dH_residual_ptr + b * stride_h_b + s * stride_h_s
    dhr0 = tl.load(dhr_base + 0 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    dhr1 = tl.load(dhr_base + 1 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    dhr2 = tl.load(dhr_base + 2 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    dhr3 = tl.load(dhr_base + 3 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)

    # dH from branch_input
    dh0 = hp0 * d_bi
    dh1 = hp1 * d_bi
    dh2 = hp2 * d_bi
    dh3 = hp3 * d_bi

    # dH from H_residual (transpose of H_res)
    dh0 += hr00 * dhr0 + hr10 * dhr1 + hr20 * dhr2 + hr30 * dhr3
    dh1 += hr01 * dhr0 + hr11 * dhr1 + hr21 * dhr2 + hr31 * dhr3
    dh2 += hr02 * dhr0 + hr12 * dhr1 + hr22 * dhr2 + hr32 * dhr3
    dh3 += hr03 * dhr0 + hr13 * dhr1 + hr23 * dhr2 + hr33 * dhr3

    # Store dH
    dh_base = dH_ptr + b * stride_h_b + s * stride_h_s
    tl.store(dh_base + 0 * stride_h_n + d_offs * stride_h_d, dh0, mask=d_mask)
    tl.store(dh_base + 1 * stride_h_n + d_offs * stride_h_d, dh1, mask=d_mask)
    tl.store(dh_base + 2 * stride_h_n + d_offs * stride_h_d, dh2, mask=d_mask)
    tl.store(dh_base + 3 * stride_h_n + d_offs * stride_h_d, dh3, mask=d_mask)

    # Partial sums for weight gradients
    num_d_blocks = tl.cdiv(dim, BLOCK_DIM)
    dhp_base = dH_pre_partial_ptr + (b * seq + s) * num_d_blocks * 4 + pid_d * 4
    tl.store(dhp_base + 0, tl.sum(tl.where(d_mask, d_bi * h0, 0.0)))
    tl.store(dhp_base + 1, tl.sum(tl.where(d_mask, d_bi * h1, 0.0)))
    tl.store(dhp_base + 2, tl.sum(tl.where(d_mask, d_bi * h2, 0.0)))
    tl.store(dhp_base + 3, tl.sum(tl.where(d_mask, d_bi * h3, 0.0)))

    # dH_res partials
    dhr_out = dH_res_partial_ptr + (b * seq + s) * num_d_blocks * 16 + pid_d * 16
    tl.store(dhr_out + 0, tl.sum(tl.where(d_mask, dhr0 * h0, 0.0)))
    tl.store(dhr_out + 1, tl.sum(tl.where(d_mask, dhr0 * h1, 0.0)))
    tl.store(dhr_out + 2, tl.sum(tl.where(d_mask, dhr0 * h2, 0.0)))
    tl.store(dhr_out + 3, tl.sum(tl.where(d_mask, dhr0 * h3, 0.0)))
    tl.store(dhr_out + 4, tl.sum(tl.where(d_mask, dhr1 * h0, 0.0)))
    tl.store(dhr_out + 5, tl.sum(tl.where(d_mask, dhr1 * h1, 0.0)))
    tl.store(dhr_out + 6, tl.sum(tl.where(d_mask, dhr1 * h2, 0.0)))
    tl.store(dhr_out + 7, tl.sum(tl.where(d_mask, dhr1 * h3, 0.0)))
    tl.store(dhr_out + 8, tl.sum(tl.where(d_mask, dhr2 * h0, 0.0)))
    tl.store(dhr_out + 9, tl.sum(tl.where(d_mask, dhr2 * h1, 0.0)))
    tl.store(dhr_out + 10, tl.sum(tl.where(d_mask, dhr2 * h2, 0.0)))
    tl.store(dhr_out + 11, tl.sum(tl.where(d_mask, dhr2 * h3, 0.0)))
    tl.store(dhr_out + 12, tl.sum(tl.where(d_mask, dhr3 * h0, 0.0)))
    tl.store(dhr_out + 13, tl.sum(tl.where(d_mask, dhr3 * h1, 0.0)))
    tl.store(dhr_out + 14, tl.sum(tl.where(d_mask, dhr3 * h2, 0.0)))
    tl.store(dhr_out + 15, tl.sum(tl.where(d_mask, dhr3 * h3, 0.0)))


@triton.jit
def _reduce_stream_mix_weights_kernel(
    dH_pre_partial_ptr, dH_res_partial_ptr, dH_pre_ptr, dH_res_ptr,
    batch, seq, num_d_blocks,
):
    """Reduce partial sums for stream mix weight gradients."""
    b = tl.program_id(0)
    if b >= batch:
        return

    # Accumulators for H_pre (4 values)
    acc_pre0, acc_pre1, acc_pre2, acc_pre3 = 0.0, 0.0, 0.0, 0.0

    # Accumulators for H_res (16 values, 4x4 matrix)
    acc_r00, acc_r01, acc_r02, acc_r03 = 0.0, 0.0, 0.0, 0.0
    acc_r10, acc_r11, acc_r12, acc_r13 = 0.0, 0.0, 0.0, 0.0
    acc_r20, acc_r21, acc_r22, acc_r23 = 0.0, 0.0, 0.0, 0.0
    acc_r30, acc_r31, acc_r32, acc_r33 = 0.0, 0.0, 0.0, 0.0

    for s in range(seq):
        for d_blk in range(num_d_blocks):
            pre_base = (b * seq + s) * num_d_blocks * 4 + d_blk * 4
            acc_pre0 += tl.load(dH_pre_partial_ptr + pre_base + 0).to(tl.float32)
            acc_pre1 += tl.load(dH_pre_partial_ptr + pre_base + 1).to(tl.float32)
            acc_pre2 += tl.load(dH_pre_partial_ptr + pre_base + 2).to(tl.float32)
            acc_pre3 += tl.load(dH_pre_partial_ptr + pre_base + 3).to(tl.float32)

            res_base = (b * seq + s) * num_d_blocks * 16 + d_blk * 16
            acc_r00 += tl.load(dH_res_partial_ptr + res_base + 0).to(tl.float32)
            acc_r01 += tl.load(dH_res_partial_ptr + res_base + 1).to(tl.float32)
            acc_r02 += tl.load(dH_res_partial_ptr + res_base + 2).to(tl.float32)
            acc_r03 += tl.load(dH_res_partial_ptr + res_base + 3).to(tl.float32)
            acc_r10 += tl.load(dH_res_partial_ptr + res_base + 4).to(tl.float32)
            acc_r11 += tl.load(dH_res_partial_ptr + res_base + 5).to(tl.float32)
            acc_r12 += tl.load(dH_res_partial_ptr + res_base + 6).to(tl.float32)
            acc_r13 += tl.load(dH_res_partial_ptr + res_base + 7).to(tl.float32)
            acc_r20 += tl.load(dH_res_partial_ptr + res_base + 8).to(tl.float32)
            acc_r21 += tl.load(dH_res_partial_ptr + res_base + 9).to(tl.float32)
            acc_r22 += tl.load(dH_res_partial_ptr + res_base + 10).to(tl.float32)
            acc_r23 += tl.load(dH_res_partial_ptr + res_base + 11).to(tl.float32)
            acc_r30 += tl.load(dH_res_partial_ptr + res_base + 12).to(tl.float32)
            acc_r31 += tl.load(dH_res_partial_ptr + res_base + 13).to(tl.float32)
            acc_r32 += tl.load(dH_res_partial_ptr + res_base + 14).to(tl.float32)
            acc_r33 += tl.load(dH_res_partial_ptr + res_base + 15).to(tl.float32)

    # Store dH_pre
    pre_out = dH_pre_ptr + b * 4
    tl.store(pre_out + 0, acc_pre0)
    tl.store(pre_out + 1, acc_pre1)
    tl.store(pre_out + 2, acc_pre2)
    tl.store(pre_out + 3, acc_pre3)

    # Store dH_res
    res_out = dH_res_ptr + b * 16
    tl.store(res_out + 0, acc_r00)
    tl.store(res_out + 1, acc_r01)
    tl.store(res_out + 2, acc_r02)
    tl.store(res_out + 3, acc_r03)
    tl.store(res_out + 4, acc_r10)
    tl.store(res_out + 5, acc_r11)
    tl.store(res_out + 6, acc_r12)
    tl.store(res_out + 7, acc_r13)
    tl.store(res_out + 8, acc_r20)
    tl.store(res_out + 9, acc_r21)
    tl.store(res_out + 10, acc_r22)
    tl.store(res_out + 11, acc_r23)
    tl.store(res_out + 12, acc_r30)
    tl.store(res_out + 13, acc_r31)
    tl.store(res_out + 14, acc_r32)
    tl.store(res_out + 15, acc_r33)


# -----------------------------------------------------------------------------
# Python Wrappers
# -----------------------------------------------------------------------------

def sinkhorn_backward(
    M_orig: torch.Tensor, dP: torch.Tensor,
    num_iters: int = 20, eps: float = 1e-8,
) -> torch.Tensor:
    """Backward through Sinkhorn-Knopp with recomputation."""
    M_orig = M_orig.contiguous()
    dP = dP.contiguous()
    batch = M_orig.shape[0]
    dM = torch.empty_like(M_orig)

    _sinkhorn_backward_kernel[(batch,)](
        M_orig, dP, dM, batch,
        num_iters=num_iters, eps=eps, NUM_STREAMS=4,
    )
    return dM


def add_residual_backward(
    dH_new: torch.Tensor, branch_output: torch.Tensor, H_post: torch.Tensor,
    block_dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward through residual addition."""
    batch, seq, num_streams, dim = dH_new.shape

    dH_new = dH_new.contiguous()
    branch_output = branch_output.contiguous()
    H_post = H_post.contiguous()

    dH_residual = torch.empty_like(dH_new)
    d_branch_output = torch.empty_like(branch_output)
    num_d_blocks = triton.cdiv(dim, block_dim)
    dH_post_partial = torch.empty(batch, seq, num_d_blocks, 4, device=dH_new.device, dtype=torch.float32)

    grid = (batch * seq, num_d_blocks)
    _add_residual_backward_kernel[grid](
        dH_new, branch_output, H_post,
        dH_residual, d_branch_output, dH_post_partial,
        batch, seq, dim,
        dH_new.stride(0), dH_new.stride(1), dH_new.stride(2), dH_new.stride(3),
        branch_output.stride(0), branch_output.stride(1), branch_output.stride(2),
        BLOCK_DIM=block_dim,
    )

    dH_post = torch.empty(batch, 4, device=dH_new.device, dtype=dH_new.dtype)
    _reduce_dH_post_kernel[(batch,)](dH_post_partial, dH_post, batch, seq, num_d_blocks)

    return dH_residual, d_branch_output, dH_post


def stream_mix_backward(
    d_branch_input: torch.Tensor, dH_residual: torch.Tensor,
    H: torch.Tensor, H_pre: torch.Tensor, H_res: torch.Tensor,
    block_dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward through stream mixing."""
    batch, seq, num_streams, dim = H.shape

    d_branch_input = d_branch_input.contiguous()
    dH_residual = dH_residual.contiguous()
    H = H.contiguous()
    H_pre = H_pre.contiguous()
    H_res = H_res.contiguous()

    dH = torch.empty_like(H)
    num_d_blocks = triton.cdiv(dim, block_dim)
    dH_pre_partial = torch.empty(batch, seq, num_d_blocks, 4, device=H.device, dtype=torch.float32)
    dH_res_partial = torch.empty(batch, seq, num_d_blocks, 16, device=H.device, dtype=torch.float32)

    grid = (batch * seq, num_d_blocks)
    _stream_mix_backward_kernel[grid](
        d_branch_input, dH_residual, H, H_pre, H_res,
        dH, dH_pre_partial, dH_res_partial,
        batch, seq, dim,
        H.stride(0), H.stride(1), H.stride(2), H.stride(3),
        d_branch_input.stride(0), d_branch_input.stride(1), d_branch_input.stride(2),
        BLOCK_DIM=block_dim,
    )

    dH_pre = torch.empty(batch, 4, device=H.device, dtype=H.dtype)
    dH_res = torch.empty(batch, 4, 4, device=H.device, dtype=H.dtype)
    _reduce_stream_mix_weights_kernel[(batch,)](
        dH_pre_partial, dH_res_partial, dH_pre, dH_res,
        batch, seq, num_d_blocks,
    )

    return dH, dH_pre, dH_res

