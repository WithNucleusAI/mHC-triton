"""Forward Triton kernels for Manifold-Constrained Hyper-Connections.

This module implements optimized GPU kernels for the mHC paper's forward pass:
https://arxiv.org/html/2512.24880

Key optimizations from Section 4.3:
- Fused operations to minimize kernel launches and memory traffic
- Transposed weight matrix layout for coalesced memory access
- Inline Sinkhorn-Knopp (eliminates separate kernel launch)
- 4x4 matrices kept entirely in registers (16 scalars per batch)
- Mixed-precision pipeline: BF16/FP16 input → FP32 compute → FP32 output

Kernel Overview:
- _sinkhorn_kernel: Project matrices to doubly stochastic
- _stream_mix_kernel: Compute weighted stream mixing (Eq. 10-11)
- _add_residual_kernel: Distribute layer output to streams (Eq. 12)
- _fused_dynamic_weights_kernel: Compute H_pre, H_post, H_res (Eq. 14-19)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# =============================================================================
# Sinkhorn-Knopp Kernel
# =============================================================================

@triton.jit
def _sinkhorn_kernel(
    M_ptr,
    batch,
    num_iters: tl.constexpr,
    eps: tl.constexpr,
    NUM_STREAMS: tl.constexpr,
):
    """
    Project matrices to doubly stochastic via Sinkhorn-Knopp iteration.
    
    Operates in-place on a batch of 4x4 matrices, alternating between
    row and column normalization until convergence.
    
    Algorithm:
        1. Apply abs() + eps to ensure positive entries
        2. For each iteration:
           - Normalize rows to sum to 1
           - Normalize columns to sum to 1
        3. Result: doubly stochastic matrix (rows and cols sum to 1)
    
    Grid: (batch,) - one program per batch element
    
    Memory Layout:
        M_ptr points to contiguous [batch, 4, 4] tensor in row-major order.
        Each 4x4 matrix is stored as 16 consecutive floats.
    
    Args:
        M_ptr: Pointer to input/output tensor [batch, 4, 4]
        batch: Batch size
        num_iters: Number of row/column normalization iterations (constexpr)
        eps: Small constant for numerical stability (constexpr)
        NUM_STREAMS: Must be 4 (constexpr, for static unrolling)
    
    Note:
        Entire 4x4 matrix is kept in 16 scalar registers for maximum performance.
        This avoids shared memory and enables efficient in-register computation.
    """
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
        # Row normalization: each row sums to 1
        r0 = m00 + m01 + m02 + m03 + eps
        r1 = m10 + m11 + m12 + m13 + eps
        r2 = m20 + m21 + m22 + m23 + eps
        r3 = m30 + m31 + m32 + m33 + eps
        m00 /= r0; m01 /= r0; m02 /= r0; m03 /= r0
        m10 /= r1; m11 /= r1; m12 /= r1; m13 /= r1
        m20 /= r2; m21 /= r2; m22 /= r2; m23 /= r2
        m30 /= r3; m31 /= r3; m32 /= r3; m33 /= r3

        # Column normalization: each column sums to 1
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


# =============================================================================
# Stream Mixing Kernel (Eq. 10-11)
# =============================================================================

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
    Fused stream mixing: compute branch_input and H_residual in one pass.
    
    Implements Eq. 10-11 from the mHC paper:
        branch_input[b,s,d] = sum_n(H_pre[b,n] * H[b,s,n,d])      (Eq. 10)
        H_residual[b,s,n,d] = sum_m(H_res[b,n,m] * H[b,s,m,d])    (Eq. 11)
    
    Grid: (batch * seq, cdiv(dim, BLOCK_DIM))
        - First axis: one program per (batch, sequence) pair
        - Second axis: tiles over hidden dimension
    
    Memory Access Pattern:
        - H: Strided access across streams (4 loads per position)
        - H_pre: Broadcast across sequence and dimension (4 scalars per batch)
        - H_res: Broadcast across sequence and dimension (16 scalars per batch)
        - Outputs: Coalesced writes along dimension
    
    Args:
        H_ptr: Input hyper-hidden [batch, seq, 4, dim]
        H_pre_ptr: Pre-mixing weights [batch, 4], normalized to sum=1
        H_res_ptr: Residual mixing matrix [batch, 4, 4], doubly stochastic
        branch_input_ptr: Output weighted sum [batch, seq, dim]
        H_residual_ptr: Output mixed residual [batch, seq, 4, dim]
        batch, seq, dim: Tensor dimensions
        stride_*: Memory strides for each tensor
        BLOCK_DIM: Tile size for dimension axis (constexpr)
    
    Performance Notes:
        - H_pre (4 values) and H_res (16 values) are loaded once per (batch, seq)
        - All 4 streams processed together to maximize register reuse
        - BLOCK_DIM=128 typically optimal for modern GPUs
    """
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bs // seq
    s = pid_bs % seq

    d_offs = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d_offs < dim

    # Load H[b,s,:,d_offs] - all 4 streams for this position
    h_base = H_ptr + b * stride_h_b + s * stride_h_s
    h0 = tl.load(h_base + 0 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h1 = tl.load(h_base + 1 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h2 = tl.load(h_base + 2 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    h3 = tl.load(h_base + 3 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)

    # Load H_pre[b,:] - pre-mixing weights (sum to 1)
    hp_base = H_pre_ptr + b * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)

    # Eq. 10: branch_input = weighted sum of streams
    branch = hp0 * h0 + hp1 * h1 + hp2 * h2 + hp3 * h3
    bi_base = branch_input_ptr + b * stride_bi_b + s * stride_bi_s
    tl.store(bi_base + d_offs * stride_bi_d, branch, mask=d_mask)

    # Load H_res[b,:,:] - 4x4 doubly stochastic matrix
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

    # Eq. 11: H_residual = H_res @ H (matrix-vector product per stream)
    out0 = hr00 * h0 + hr01 * h1 + hr02 * h2 + hr03 * h3
    out1 = hr10 * h0 + hr11 * h1 + hr12 * h2 + hr13 * h3
    out2 = hr20 * h0 + hr21 * h1 + hr22 * h2 + hr23 * h3
    out3 = hr30 * h0 + hr31 * h1 + hr32 * h2 + hr33 * h3

    out_base = H_residual_ptr + b * stride_hr_b + s * stride_hr_s
    tl.store(out_base + 0 * stride_hr_n + d_offs * stride_hr_d, out0, mask=d_mask)
    tl.store(out_base + 1 * stride_hr_n + d_offs * stride_hr_d, out1, mask=d_mask)
    tl.store(out_base + 2 * stride_hr_n + d_offs * stride_hr_d, out2, mask=d_mask)
    tl.store(out_base + 3 * stride_hr_n + d_offs * stride_hr_d, out3, mask=d_mask)


# =============================================================================
# Add Residual Kernel (Eq. 12)
# Combines layer output with residual streams
# =============================================================================

@triton.jit
def _add_residual_kernel(
    H_residual_ptr, branch_output_ptr, H_post_ptr, H_new_ptr,
    batch, seq, dim,
    stride_h_b, stride_h_s, stride_h_n, stride_h_d,
    stride_bo_b, stride_bo_s, stride_bo_d,
    BLOCK_DIM: tl.constexpr,
):
    """
    Distribute layer output to streams and add to residual.
    
    Implements Eq. 12 from the mHC paper:
        H_new[b,s,n,d] = H_residual[b,s,n,d] + H_post[b,n] * branch_output[b,s,d]
    
    This distributes the layer's output back to all streams, weighted by H_post,
    then adds the residual connection.
    
    Grid: (batch * seq, cdiv(dim, BLOCK_DIM))
        - First axis: one program per (batch, sequence) pair
        - Second axis: tiles over hidden dimension
    
    Args:
        H_residual_ptr: Residual streams [batch, seq, 4, dim] from stream_mix
        branch_output_ptr: Layer output [batch, seq, dim]
        H_post_ptr: Post-distribution weights [batch, 4], values in (0, 2)
        H_new_ptr: Output updated hyper-hidden [batch, seq, 4, dim]
        batch, seq, dim: Tensor dimensions
        stride_*: Memory strides for each tensor
        BLOCK_DIM: Tile size for dimension axis (constexpr)
    
    Note:
        H_post values are in (0, 2) from 2*sigmoid activation, allowing the
        layer output to be amplified or attenuated per stream.
    """
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bs // seq
    s = pid_bs % seq

    d_offs = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    d_mask = d_offs < dim

    # Load branch_output[b,s,d_offs]
    bo_base = branch_output_ptr + b * stride_bo_b + s * stride_bo_s
    branch = tl.load(bo_base + d_offs * stride_bo_d, mask=d_mask, other=0.0).to(tl.float32)

    # Load H_post[b,:] - post-distribution weights
    hp_base = H_post_ptr + b * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)

    hr_base = H_residual_ptr + b * stride_h_b + s * stride_h_s
    hn_base = H_new_ptr + b * stride_h_b + s * stride_h_s

    # Eq. 12: H_new[n] = H_residual[n] + H_post[n] * branch_output
    hr0 = tl.load(hr_base + 0 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    hr1 = tl.load(hr_base + 1 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    hr2 = tl.load(hr_base + 2 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)
    hr3 = tl.load(hr_base + 3 * stride_h_n + d_offs * stride_h_d, mask=d_mask, other=0.0).to(tl.float32)

    tl.store(hn_base + 0 * stride_h_n + d_offs * stride_h_d, hr0 + hp0 * branch, mask=d_mask)
    tl.store(hn_base + 1 * stride_h_n + d_offs * stride_h_d, hr1 + hp1 * branch, mask=d_mask)
    tl.store(hn_base + 2 * stride_h_n + d_offs * stride_h_d, hr2 + hp2 * branch, mask=d_mask)
    tl.store(hn_base + 3 * stride_h_n + d_offs * stride_h_d, hr3 + hp3 * branch, mask=d_mask)


# =============================================================================
# Fused Dynamic Weight Kernel (Eq. 14-19)
# Fully fused: matmul + RMS norm + activations + Sinkhorn
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 256}, num_warps=4),
        triton.Config({'BLOCK_K': 512}, num_warps=4),
        triton.Config({'BLOCK_K': 1024}, num_warps=8),
        triton.Config({'BLOCK_K': 2048}, num_warps=8),
    ],
    key=['in_dim'],
)
@triton.jit
def _fused_dynamic_weights_kernel(
    # Inputs
    x_ptr,              # [batch, in_dim] - flattened hyper-hidden (mean pooled)
    phi_t_ptr,          # [out_dim, in_dim] - TRANSPOSED for coalesced access
    bias_ptr,           # [out_dim] - combined bias (base params)
    alpha_pre,          # scalar - scale for H_pre
    alpha_post,         # scalar - scale for H_post
    alpha_res,          # scalar - scale for H_res
    # Outputs
    H_pre_ptr,          # [batch, n] - pre-mixing weights
    H_post_ptr,         # [batch, n] - post-distribution weights
    H_res_ptr,          # [batch, n, n] - residual matrix (post-Sinkhorn)
    # Dimensions
    batch,
    in_dim,             # nC (e.g., 4*4096 = 16384)
    # Constants
    BLOCK_K: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fully fused kernel implementing Eq. 14-19 with inline Sinkhorn.
    
    This kernel computes all three dynamic connection weights (H_pre, H_post, H_res)
    from the mean-pooled hyper-hidden state in a single pass, including the
    Sinkhorn-Knopp projection for H_res.
    
    Mathematical Operations (from paper):
        Eq. 14: raw = x @ phi                      (linear projection)
        Eq. 15: r = ||x||_2 / sqrt(in_dim)         (RMS norm factor)
        Eq. 16: scaled = (1/r) * alpha * raw + bias (scale and shift)
        Eq. 17: H_pre = normalize(sigmoid(scaled[:n]))
        Eq. 18: H_post = 2 * sigmoid(scaled[n:2n])
        Eq. 19: H_res = Sinkhorn(scaled[2n:])
    
    Key Optimizations:
        1. Transposed phi layout [24, in_dim] for coalesced memory reads
        2. Fused RMS norm computation during matmul (single pass over x)
        3. Inline Sinkhorn-Knopp (no separate kernel launch)
        4. 24 scalar accumulators fit in registers
    
    Grid: (batch,) - one program per batch element
    
    Memory Access:
        - x: Sequential read in BLOCK_K chunks (coalesced)
        - phi_t: 24 coalesced row reads per chunk (each row is contiguous)
        - bias: Single read of 24 values at kernel end
        - Output: 24 scalar writes per batch element
    
    Args:
        x_ptr: Mean-pooled input [batch, n*dim], contiguous
        phi_t_ptr: Transposed projection [24, n*dim] for coalesced access
        bias_ptr: Combined bias [24] = [H_pre_base, H_post_base, H_res_base.flatten()]
        alpha_pre/post/res: Learnable scaling factors for each output group
        H_pre_ptr: Output [batch, 4], normalized to sum=1
        H_post_ptr: Output [batch, 4], values in (0, 2)
        H_res_ptr: Output [batch, 4, 4], doubly stochastic
        batch: Batch size
        in_dim: Input dimension (n * hidden_dim)
        BLOCK_K: Tile size for reduction (auto-tuned)
        sinkhorn_iters: Number of Sinkhorn iterations (typically 20)
        eps: Numerical stability constant
    """
    pid = tl.program_id(0)
    if pid >= batch:
        return
    
    # Constants
    OUT_DIM: tl.constexpr = 24  # n^2 + 2n for n=4
    N: tl.constexpr = 4
    
    # Initialize 24 accumulators for matmul output
    acc0 = 0.0; acc1 = 0.0; acc2 = 0.0; acc3 = 0.0
    acc4 = 0.0; acc5 = 0.0; acc6 = 0.0; acc7 = 0.0
    acc8 = 0.0; acc9 = 0.0; acc10 = 0.0; acc11 = 0.0
    acc12 = 0.0; acc13 = 0.0; acc14 = 0.0; acc15 = 0.0
    acc16 = 0.0; acc17 = 0.0; acc18 = 0.0; acc19 = 0.0
    acc20 = 0.0; acc21 = 0.0; acc22 = 0.0; acc23 = 0.0
    norm_sq = 0.0  # For RMS norm: sum(x^2)
    
    # Base pointer for this batch element
    x_base = x_ptr + pid * in_dim
    
    # Stream through input dimension with coalesced phi access
    # This computes both the matmul and the norm in a single pass
    for k_start in range(0, in_dim, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < in_dim
        
        # Load x block (contiguous read)
        x_vals = tl.load(x_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Accumulate squared norm for RMS
        norm_sq += tl.sum(x_vals * x_vals)
        
        # Load phi_t rows (COALESCED: each row [i,:] is contiguous in memory)
        # phi_t layout: [24, in_dim] where row i contains phi[:, i].T
        phi0 = tl.load(phi_t_ptr + 0 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi1 = tl.load(phi_t_ptr + 1 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi2 = tl.load(phi_t_ptr + 2 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi3 = tl.load(phi_t_ptr + 3 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi4 = tl.load(phi_t_ptr + 4 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi5 = tl.load(phi_t_ptr + 5 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi6 = tl.load(phi_t_ptr + 6 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi7 = tl.load(phi_t_ptr + 7 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi8 = tl.load(phi_t_ptr + 8 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi9 = tl.load(phi_t_ptr + 9 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi10 = tl.load(phi_t_ptr + 10 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi11 = tl.load(phi_t_ptr + 11 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi12 = tl.load(phi_t_ptr + 12 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi13 = tl.load(phi_t_ptr + 13 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi14 = tl.load(phi_t_ptr + 14 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi15 = tl.load(phi_t_ptr + 15 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi16 = tl.load(phi_t_ptr + 16 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi17 = tl.load(phi_t_ptr + 17 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi18 = tl.load(phi_t_ptr + 18 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi19 = tl.load(phi_t_ptr + 19 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi20 = tl.load(phi_t_ptr + 20 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi21 = tl.load(phi_t_ptr + 21 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi22 = tl.load(phi_t_ptr + 22 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        phi23 = tl.load(phi_t_ptr + 23 * in_dim + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Accumulate dot products (Eq. 14: raw = x @ phi)
        acc0 += tl.sum(x_vals * phi0)
        acc1 += tl.sum(x_vals * phi1)
        acc2 += tl.sum(x_vals * phi2)
        acc3 += tl.sum(x_vals * phi3)
        acc4 += tl.sum(x_vals * phi4)
        acc5 += tl.sum(x_vals * phi5)
        acc6 += tl.sum(x_vals * phi6)
        acc7 += tl.sum(x_vals * phi7)
        acc8 += tl.sum(x_vals * phi8)
        acc9 += tl.sum(x_vals * phi9)
        acc10 += tl.sum(x_vals * phi10)
        acc11 += tl.sum(x_vals * phi11)
        acc12 += tl.sum(x_vals * phi12)
        acc13 += tl.sum(x_vals * phi13)
        acc14 += tl.sum(x_vals * phi14)
        acc15 += tl.sum(x_vals * phi15)
        acc16 += tl.sum(x_vals * phi16)
        acc17 += tl.sum(x_vals * phi17)
        acc18 += tl.sum(x_vals * phi18)
        acc19 += tl.sum(x_vals * phi19)
        acc20 += tl.sum(x_vals * phi20)
        acc21 += tl.sum(x_vals * phi21)
        acc22 += tl.sum(x_vals * phi22)
        acc23 += tl.sum(x_vals * phi23)
    
    # Eq. 15-16: Compute inverse RMS and apply scaling
    # RMSNorm: x_norm = x / sqrt(mean(x^2) + eps)
    # We reorder: scaled = (1/r) * alpha * raw + bias where r = sqrt(norm_sq/dim + eps)
    inv_rms = 1.0 / tl.sqrt(norm_sq / in_dim + eps)
    
    # Load biases (base parameters)
    b0 = tl.load(bias_ptr + 0); b1 = tl.load(bias_ptr + 1)
    b2 = tl.load(bias_ptr + 2); b3 = tl.load(bias_ptr + 3)
    b4 = tl.load(bias_ptr + 4); b5 = tl.load(bias_ptr + 5)
    b6 = tl.load(bias_ptr + 6); b7 = tl.load(bias_ptr + 7)
    b8 = tl.load(bias_ptr + 8); b9 = tl.load(bias_ptr + 9)
    b10 = tl.load(bias_ptr + 10); b11 = tl.load(bias_ptr + 11)
    b12 = tl.load(bias_ptr + 12); b13 = tl.load(bias_ptr + 13)
    b14 = tl.load(bias_ptr + 14); b15 = tl.load(bias_ptr + 15)
    b16 = tl.load(bias_ptr + 16); b17 = tl.load(bias_ptr + 17)
    b18 = tl.load(bias_ptr + 18); b19 = tl.load(bias_ptr + 19)
    b20 = tl.load(bias_ptr + 20); b21 = tl.load(bias_ptr + 21)
    b22 = tl.load(bias_ptr + 22); b23 = tl.load(bias_ptr + 23)
    
    # Eq. 16: Apply scaling - scaled = inv_rms * alpha * acc + bias
    # H_pre uses indices 0-3, H_post uses 4-7, H_res uses 8-23
    s0 = inv_rms * alpha_pre * acc0 + b0
    s1 = inv_rms * alpha_pre * acc1 + b1
    s2 = inv_rms * alpha_pre * acc2 + b2
    s3 = inv_rms * alpha_pre * acc3 + b3
    
    s4 = inv_rms * alpha_post * acc4 + b4
    s5 = inv_rms * alpha_post * acc5 + b5
    s6 = inv_rms * alpha_post * acc6 + b6
    s7 = inv_rms * alpha_post * acc7 + b7
    
    s8 = inv_rms * alpha_res * acc8 + b8
    s9 = inv_rms * alpha_res * acc9 + b9
    s10 = inv_rms * alpha_res * acc10 + b10
    s11 = inv_rms * alpha_res * acc11 + b11
    s12 = inv_rms * alpha_res * acc12 + b12
    s13 = inv_rms * alpha_res * acc13 + b13
    s14 = inv_rms * alpha_res * acc14 + b14
    s15 = inv_rms * alpha_res * acc15 + b15
    s16 = inv_rms * alpha_res * acc16 + b16
    s17 = inv_rms * alpha_res * acc17 + b17
    s18 = inv_rms * alpha_res * acc18 + b18
    s19 = inv_rms * alpha_res * acc19 + b19
    s20 = inv_rms * alpha_res * acc20 + b20
    s21 = inv_rms * alpha_res * acc21 + b21
    s22 = inv_rms * alpha_res * acc22 + b22
    s23 = inv_rms * alpha_res * acc23 + b23
    
    # Eq. 17: H_pre = normalize(sigmoid(s0:s3))
    # Pre-mixing weights sum to 1
    sig0 = tl.sigmoid(s0); sig1 = tl.sigmoid(s1)
    sig2 = tl.sigmoid(s2); sig3 = tl.sigmoid(s3)
    pre_sum = sig0 + sig1 + sig2 + sig3 + eps
    hp0 = sig0 / pre_sum; hp1 = sig1 / pre_sum
    hp2 = sig2 / pre_sum; hp3 = sig3 / pre_sum
    
    # Eq. 18: H_post = 2 * sigmoid(s4:s7)
    # Post-distribution weights in (0, 2)
    hpost0 = 2.0 * tl.sigmoid(s4); hpost1 = 2.0 * tl.sigmoid(s5)
    hpost2 = 2.0 * tl.sigmoid(s6); hpost3 = 2.0 * tl.sigmoid(s7)
    
    # Store H_pre
    pre_base = H_pre_ptr + pid * N
    tl.store(pre_base + 0, hp0); tl.store(pre_base + 1, hp1)
    tl.store(pre_base + 2, hp2); tl.store(pre_base + 3, hp3)
    
    # Store H_post
    post_base = H_post_ptr + pid * N
    tl.store(post_base + 0, hpost0); tl.store(post_base + 1, hpost1)
    tl.store(post_base + 2, hpost2); tl.store(post_base + 3, hpost3)
    
    # Eq. 19: H_res = Sinkhorn(s8:s23)
    # Inline Sinkhorn-Knopp projection to doubly stochastic matrix
    # Apply abs() + eps to ensure positive entries
    m00 = tl.abs(s8) + eps;  m01 = tl.abs(s9) + eps
    m02 = tl.abs(s10) + eps; m03 = tl.abs(s11) + eps
    m10 = tl.abs(s12) + eps; m11 = tl.abs(s13) + eps
    m12 = tl.abs(s14) + eps; m13 = tl.abs(s15) + eps
    m20 = tl.abs(s16) + eps; m21 = tl.abs(s17) + eps
    m22 = tl.abs(s18) + eps; m23 = tl.abs(s19) + eps
    m30 = tl.abs(s20) + eps; m31 = tl.abs(s21) + eps
    m32 = tl.abs(s22) + eps; m33 = tl.abs(s23) + eps
    
    # Alternating row/column normalization (Sinkhorn iterations)
    for _ in range(sinkhorn_iters):
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
    
    # Store H_res (doubly stochastic: rows and columns sum to 1)
    res_base = H_res_ptr + pid * 16
    tl.store(res_base + 0, m00); tl.store(res_base + 1, m01)
    tl.store(res_base + 2, m02); tl.store(res_base + 3, m03)
    tl.store(res_base + 4, m10); tl.store(res_base + 5, m11)
    tl.store(res_base + 6, m12); tl.store(res_base + 7, m13)
    tl.store(res_base + 8, m20); tl.store(res_base + 9, m21)
    tl.store(res_base + 10, m22); tl.store(res_base + 11, m23)
    tl.store(res_base + 12, m30); tl.store(res_base + 13, m31)
    tl.store(res_base + 14, m32); tl.store(res_base + 15, m33)


# =============================================================================
# Python Wrappers
# =============================================================================

def sinkhorn_forward(M: torch.Tensor, num_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """
    Project matrices to doubly stochastic via Sinkhorn-Knopp.
    
    Args:
        M: Input tensor [batch, 4, 4]
        num_iters: Number of row/column normalization iterations
        eps: Small constant for numerical stability
        
    Returns:
        Doubly stochastic tensor [batch, 4, 4] where all rows and columns sum to 1
    """
    assert M.shape[-2:] == (4, 4), "Expected (..., 4, 4)"
    batch = M.shape[0]

    M_out = M.contiguous().clone()
    _sinkhorn_kernel[(batch,)](M_out, batch, num_iters=num_iters, eps=eps, NUM_STREAMS=4)
    return M_out


def stream_mix_forward(
    H: torch.Tensor, H_pre: torch.Tensor, H_res: torch.Tensor, block_dim: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute stream mixing (Eq. 10-11).
    
    Args:
        H: Hyper-hidden state [batch, seq, 4, dim]
        H_pre: Pre-mixing weights [batch, 4], should sum to 1
        H_res: Residual mixing matrix [batch, 4, 4], doubly stochastic
        block_dim: Tile size for dimension axis
        
    Returns:
        branch_input: Weighted sum of streams [batch, seq, dim]
        H_residual: Mixed residual streams [batch, seq, 4, dim]
    """
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
    """
    Distribute layer output to streams and add residual (Eq. 12).
    
    Args:
        H_residual: Residual streams [batch, seq, 4, dim]
        branch_output: Layer output [batch, seq, dim]
        H_post: Post-distribution weights [batch, 4], values in (0, 2)
        block_dim: Tile size for dimension axis
        
    Returns:
        H_new: Updated hyper-hidden [batch, seq, 4, dim]
    """
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


def fused_dynamic_weights_forward(
    x: torch.Tensor,
    phi: torch.Tensor,
    bias: torch.Tensor,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute dynamic connection weights with fused kernel (Eq. 14-19).
    
    This is the main entry point for computing input-dependent mixing weights.
    All operations (matmul, RMS norm, activations, Sinkhorn) are fused into
    a single kernel launch for maximum efficiency.
    
    Args:
        x: Mean-pooled input [batch, n*dim], typically H.mean(dim=1).flatten(-2)
        phi: Projection matrix [n*dim, 24]
        bias: Combined bias [24] = [H_pre_base, H_post_base, H_res_base.flatten()]
        alpha_pre: Learnable scale for H_pre projection
        alpha_post: Learnable scale for H_post projection
        alpha_res: Learnable scale for H_res projection
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations (default 20)
        eps: Numerical stability constant
        
    Returns:
        H_pre: Pre-mixing weights [batch, 4], normalized to sum=1
        H_post: Post-distribution weights [batch, 4], values in (0, 2)
        H_res: Residual mixing matrix [batch, 4, 4], doubly stochastic
    """
    batch, in_dim = x.shape
    n = 4  # num_streams
    out_dim = n * n + 2 * n  # 24
    
    assert phi.shape == (in_dim, out_dim), f"phi shape mismatch: {phi.shape} vs ({in_dim}, {out_dim})"
    assert bias.shape == (out_dim,), f"bias shape mismatch: {bias.shape} vs ({out_dim},)"
    
    # Ensure contiguous and float32 for computation
    x = x.contiguous().float()
    bias = bias.contiguous().float()
    
    # Transpose phi for coalesced access: [in_dim, 24] -> [24, in_dim]
    phi_t = phi.T.contiguous().float()
    
    # Allocate outputs
    H_pre = torch.empty(batch, n, device=x.device, dtype=torch.float32)
    H_post = torch.empty(batch, n, device=x.device, dtype=torch.float32)
    H_res = torch.empty(batch, n, n, device=x.device, dtype=torch.float32)
    
    # Launch fully fused kernel
    grid = (batch,)
    _fused_dynamic_weights_kernel[grid](
        x, phi_t, bias,
        alpha_pre, alpha_post, alpha_res,
        H_pre, H_post, H_res,
        batch, in_dim,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )
    
    return H_pre.contiguous(), H_post.contiguous(), H_res.contiguous()
