"""Forward Triton kernels for Manifold-Constrained Hyper-Connections.

Optimizations:
- Fused operations to minimize kernel launches
- 4x4 matrices kept in registers
- FP16 compute with FP32 accumulation
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# -----------------------------------------------------------------------------
# Sinkhorn-Knopp Kernel
# Projects matrices to doubly stochastic (all rows/cols sum to 1)
# -----------------------------------------------------------------------------

@triton.jit
def _sinkhorn_kernel(
    M_ptr,
    batch,
    num_iters: tl.constexpr,
    eps: tl.constexpr,
    NUM_STREAMS: tl.constexpr,
):
    """Projects (batch, 4, 4) matrices to doubly stochastic in-place."""
    pid = tl.program_id(0)
    if pid >= batch:
        return

    base = pid * 16

    # Load 4x4 matrix to registers, apply abs + eps
    m00 = tl.abs(tl.load(M_ptr + base + 0).to(tl.float32)) + eps
    m01 = tl.abs(tl.load(M_ptr + base + 1).to(tl.float32)) + eps
    m02 = tl.abs(tl.load(M_ptr + base + 2).to(tl.float32)) + eps
    m03 = tl.abs(tl.load(M_ptr + base + 3).to(tl.float32)) + eps
    m10 = tl.abs(tl.load(M_ptr + base + 4).to(tl.float32)) + eps
    m11 = tl.abs(tl.load(M_ptr + base + 5).to(tl.float32)) + eps
    m12 = tl.abs(tl.load(M_ptr + base + 6).to(tl.float32)) + eps
    m13 = tl.abs(tl.load(M_ptr + base + 7).to(tl.float32)) + eps
    m20 = tl.abs(tl.load(M_ptr + base + 8).to(tl.float32)) + eps
    m21 = tl.abs(tl.load(M_ptr + base + 9).to(tl.float32)) + eps
    m22 = tl.abs(tl.load(M_ptr + base + 10).to(tl.float32)) + eps
    m23 = tl.abs(tl.load(M_ptr + base + 11).to(tl.float32)) + eps
    m30 = tl.abs(tl.load(M_ptr + base + 12).to(tl.float32)) + eps
    m31 = tl.abs(tl.load(M_ptr + base + 13).to(tl.float32)) + eps
    m32 = tl.abs(tl.load(M_ptr + base + 14).to(tl.float32)) + eps
    m33 = tl.abs(tl.load(M_ptr + base + 15).to(tl.float32)) + eps

    # Alternating row/column normalization
    for _ in range(num_iters):
        # Row normalization
        r0 = m00 + m01 + m02 + m03 + eps
        r1 = m10 + m11 + m12 + m13 + eps
        r2 = m20 + m21 + m22 + m23 + eps
        r3 = m30 + m31 + m32 + m33 + eps
        m00 /= r0; m01 /= r0; m02 /= r0; m03 /= r0
        m10 /= r1; m11 /= r1; m12 /= r1; m13 /= r1
        m20 /= r2; m21 /= r2; m22 /= r2; m23 /= r2
        m30 /= r3; m31 /= r3; m32 /= r3; m33 /= r3

        # Column normalization
        c0 = m00 + m10 + m20 + m30 + eps
        c1 = m01 + m11 + m21 + m31 + eps
        c2 = m02 + m12 + m22 + m32 + eps
        c3 = m03 + m13 + m23 + m33 + eps
        m00 /= c0; m10 /= c0; m20 /= c0; m30 /= c0
        m01 /= c1; m11 /= c1; m21 /= c1; m31 /= c1
        m02 /= c2; m12 /= c2; m22 /= c2; m32 /= c2
        m03 /= c3; m13 /= c3; m23 /= c3; m33 /= c3

    # Store result
    tl.store(M_ptr + base + 0, m00)
    tl.store(M_ptr + base + 1, m01)
    tl.store(M_ptr + base + 2, m02)
    tl.store(M_ptr + base + 3, m03)
    tl.store(M_ptr + base + 4, m10)
    tl.store(M_ptr + base + 5, m11)
    tl.store(M_ptr + base + 6, m12)
    tl.store(M_ptr + base + 7, m13)
    tl.store(M_ptr + base + 8, m20)
    tl.store(M_ptr + base + 9, m21)
    tl.store(M_ptr + base + 10, m22)
    tl.store(M_ptr + base + 11, m23)
    tl.store(M_ptr + base + 12, m30)
    tl.store(M_ptr + base + 13, m31)
    tl.store(M_ptr + base + 14, m32)
    tl.store(M_ptr + base + 15, m33)


# -----------------------------------------------------------------------------
# Stream Mixing Kernel
# Computes branch_input and H_residual in a single pass
# -----------------------------------------------------------------------------

@triton.jit
def _stream_mix_kernel(
    H_ptr, H_pre_ptr, H_res_ptr,
    branch_input_ptr, H_residual_ptr,
    batch, seq, dim,
    stride_h_b, stride_h_s, stride_h_n, stride_h_d,
    stride_bi_b, stride_bi_s, stride_bi_d,
    stride_hr_b, stride_hr_s, stride_hr_n, stride_hr_d,
    BLOCK_DIM: tl.constexpr,
):
    """
    Computes:
    - branch_input = einsum('bn,bsnd->bsd', H_pre, H)
    - H_residual = einsum('bnm,bsmd->bsnd', H_res, H)
    """
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bs // seq
    s = pid_bs % seq

    d_offs = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d_offs < dim

    # Load H[b,s,:,d_offs]
    h_base = H_ptr + b * stride_h_b + s * stride_h_s
    h0 = tl.load(h_base + 0 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h1 = tl.load(h_base + 1 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h2 = tl.load(h_base + 2 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h3 = tl.load(h_base + 3 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)

    # Load H_pre[b,:]
    hp_base = H_pre_ptr + b * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)

    # branch_input = weighted sum of streams
    branch = hp0 * h0 + hp1 * h1 + hp2 * h2 + hp3 * h3
    bi_base = branch_input_ptr + b * stride_bi_b + s * stride_bi_s
    tl.store(bi_base + d_offs * stride_bi_d, branch, mask=d_mask)

    # Load H_res[b,:,:] (4x4 matrix)
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

    # H_residual = H_res @ H (matrix-vector per stream)
    out0 = hr00 * h0 + hr01 * h1 + hr02 * h2 + hr03 * h3
    out1 = hr10 * h0 + hr11 * h1 + hr12 * h2 + hr13 * h3
    out2 = hr20 * h0 + hr21 * h1 + hr22 * h2 + hr23 * h3
    out3 = hr30 * h0 + hr31 * h1 + hr32 * h2 + hr33 * h3

    out_base = H_residual_ptr + b * stride_hr_b + s * stride_hr_s
    tl.store(out_base + 0 * stride_hr_n + d_offs * stride_hr_d, out0, mask=d_mask)
    tl.store(out_base + 1 * stride_hr_n + d_offs * stride_hr_d, out1, mask=d_mask)
    tl.store(out_base + 2 * stride_hr_n + d_offs * stride_hr_d, out2, mask=d_mask)
    tl.store(out_base + 3 * stride_hr_n + d_offs * stride_hr_d, out3, mask=d_mask)


# -----------------------------------------------------------------------------
# Add Residual Kernel
# Combines layer output with residual streams
# -----------------------------------------------------------------------------

@triton.jit
def _add_residual_kernel(
    H_residual_ptr, branch_output_ptr, H_post_ptr, H_new_ptr,
    batch, seq, dim,
    stride_h_b, stride_h_s, stride_h_n, stride_h_d,
    stride_bo_b, stride_bo_s, stride_bo_d,
    BLOCK_DIM: tl.constexpr,
):
    """H_new[n] = H_residual[n] + H_post[n] * branch_output"""
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bs // seq
    s = pid_bs % seq

    d_offs = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d_offs < dim

    # Load branch_output
    bo_base = branch_output_ptr + b * stride_bo_b + s * stride_bo_s
    branch = tl.load(bo_base + d_offs * stride_bo_d, mask=d_mask, other=0.0).to(tl.float32)

    # Load H_post[b,:]
    hp_base = H_post_ptr + b * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)

    hr_base = H_residual_ptr + b * stride_h_b + s * stride_h_s
    hn_base = H_new_ptr + b * stride_h_b + s * stride_h_s

    # H_new[n] = H_residual[n] + H_post[n] * branch_output
    hr0 = tl.load(hr_base + 0 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    hr1 = tl.load(hr_base + 1 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    hr2 = tl.load(hr_base + 2 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    hr3 = tl.load(hr_base + 3 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)

    tl.store(hn_base + 0 * stride_h_n + d_offs * stride_h_d, hr0 + hp0 * branch, mask=d_mask)
    tl.store(hn_base + 1 * stride_h_n + d_offs * stride_h_d, hr1 + hp1 * branch, mask=d_mask)
    tl.store(hn_base + 2 * stride_h_n + d_offs * stride_h_d, hr2 + hp2 * branch, mask=d_mask)
    tl.store(hn_base + 3 * stride_h_n + d_offs * stride_h_d, hr3 + hp3 * branch, mask=d_mask)


# -----------------------------------------------------------------------------
# Python Wrappers
# -----------------------------------------------------------------------------

def sinkhorn_forward(M: torch.Tensor, num_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Forward-only Sinkhorn-Knopp projection."""
    assert M.shape[-2:] == (4, 4), "Expected (..., 4, 4)"
    batch = M.shape[0]

    M_out = M.contiguous().clone()
    _sinkhorn_kernel[(batch,)](M_out, batch, num_iters=num_iters, eps=eps, NUM_STREAMS=4)
    return M_out


def stream_mix_forward(
    H: torch.Tensor, H_pre: torch.Tensor, H_res: torch.Tensor, block_dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-only stream mixing."""
    batch, seq, num_streams, dim = H.shape
    assert num_streams == 4

    H = H.contiguous()
    H_pre = H_pre.contiguous()
    H_res = H_res.contiguous()

    branch_input = torch.empty(batch, seq, dim, device=H.device, dtype=H.dtype)
    H_residual = torch.empty_like(H)

    grid = (batch * seq, triton.cdiv(dim, block_dim))
    _stream_mix_kernel[grid](
        H, H_pre, H_res, branch_input, H_residual,
        batch, seq, dim,
        H.stride(0), H.stride(1), H.stride(2), H.stride(3),
        branch_input.stride(0), branch_input.stride(1), branch_input.stride(2),
        H_residual.stride(0), H_residual.stride(1), H_residual.stride(2), H_residual.stride(3),
        BLOCK_DIM=block_dim,
    )
    return branch_input, H_residual


def add_residual_forward(
    H_residual: torch.Tensor, branch_output: torch.Tensor, H_post: torch.Tensor, block_dim: int = 128,
) -> torch.Tensor:
    """Forward-only residual addition."""
    batch, seq, num_streams, dim = H_residual.shape
    assert num_streams == 4

    H_residual = H_residual.contiguous()
    branch_output = branch_output.contiguous()
    H_post = H_post.contiguous()

    H_new = torch.empty_like(H_residual)
    grid = (batch * seq, triton.cdiv(dim, block_dim))

    _add_residual_kernel[grid](
        H_residual, branch_output, H_post, H_new,
        batch, seq, dim,
        H_residual.stride(0), H_residual.stride(1), H_residual.stride(2), H_residual.stride(3),
        branch_output.stride(0), branch_output.stride(1), branch_output.stride(2),
        BLOCK_DIM=block_dim,
    )
    return H_new

