"""High-level ops with autograd support for Hyper-Connections."""

import torch
from typing import Tuple

from ._kernels import (
    sinkhorn_forward, stream_mix_forward, add_residual_forward,
    fused_dynamic_weights_forward,
)
from ._backward import (
    sinkhorn_backward, stream_mix_backward, add_residual_backward,
)


class _SinkhornKnopp(torch.autograd.Function):
    """Sinkhorn-Knopp projection with fused backward."""

    @staticmethod
    def forward(ctx, M: torch.Tensor, num_iters: int, eps: float) -> torch.Tensor:
        ctx.save_for_backward(M)
        ctx.num_iters = num_iters
        ctx.eps = eps
        return sinkhorn_forward(M, num_iters, eps)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        M, = ctx.saved_tensors
        grad_M = sinkhorn_backward(M, grad_output, ctx.num_iters, ctx.eps)
        return grad_M, None, None


class _FusedStreamMix(torch.autograd.Function):
    """Fused stream mixing with Triton backward."""

    @staticmethod
    def forward(ctx, H: torch.Tensor, H_pre: torch.Tensor, H_res: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_input, H_residual = stream_mix_forward(H, H_pre, H_res)
        ctx.save_for_backward(H, H_pre, H_res)
        return branch_input, H_residual

    @staticmethod
    def backward(ctx, grad_branch_input: torch.Tensor, grad_H_residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, H_pre, H_res = ctx.saved_tensors
        grad_H, grad_H_pre, grad_H_res = stream_mix_backward(
            grad_branch_input, grad_H_residual, H, H_pre, H_res
        )
        return grad_H, grad_H_pre, grad_H_res


class _FusedAddResidual(torch.autograd.Function):
    """Fused add residual with Triton backward."""

    @staticmethod
    def forward(ctx, H_residual: torch.Tensor, branch_output: torch.Tensor, H_post: torch.Tensor
    ) -> torch.Tensor:
        H_new = add_residual_forward(H_residual, branch_output, H_post)
        ctx.save_for_backward(branch_output, H_post)
        return H_new

    @staticmethod
    def backward(ctx, grad_H_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_output, H_post = ctx.saved_tensors
        grad_H_residual, grad_branch_output, grad_H_post = add_residual_backward(
            grad_H_new, branch_output, H_post
        )
        return grad_H_residual, grad_branch_output, grad_H_post


class _FusedDynamicWeights(torch.autograd.Function):
    """
    Fused dynamic weight computation with autograd support.
    
    Implements Eq. 14-19 from the paper in a single fused forward kernel:
    - Fused matmul + RMS norm computation
    - Fused scaling + bias + activation
    - Inline Sinkhorn-Knopp projection (no separate kernel)
    
    Backward uses PyTorch for matmuls and Triton for Sinkhorn.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        phi: torch.Tensor,
        bias: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        sinkhorn_iters: int,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_pre, H_post, H_res = fused_dynamic_weights_forward(
            x, phi, bias,
            alpha_pre.item(), alpha_post.item(), alpha_res.item(),
            sinkhorn_iters, eps,
        )
        
        # Save for backward
        ctx.save_for_backward(x, phi, bias, alpha_pre, alpha_post, alpha_res, H_pre, H_post, H_res)
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps
        
        return H_pre, H_post, H_res

    @staticmethod
    def backward(
        ctx, grad_H_pre: torch.Tensor, grad_H_post: torch.Tensor, grad_H_res: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        x, phi, bias, alpha_pre, alpha_post, alpha_res, H_pre, H_post, H_res = ctx.saved_tensors
        
        batch, in_dim = x.shape
        n = 4
        
        # Store original dtypes for output gradients
        x_dtype = x.dtype
        phi_dtype = phi.dtype
        bias_dtype = bias.dtype
        
        # Convert everything to float32 for computation
        x = x.float()
        phi = phi.float()
        bias = bias.float()
        alpha_pre = alpha_pre.float()
        alpha_post = alpha_post.float()
        alpha_res = alpha_res.float()
        H_pre = H_pre.float()
        grad_H_pre = grad_H_pre.float()
        grad_H_post = grad_H_post.float()
        grad_H_res = grad_H_res.float()
        
        # Recompute intermediate values for backward
        # RMS norm factor
        rms = torch.sqrt((x * x).sum(dim=-1, keepdim=True) / in_dim + ctx.eps)
        inv_rms = 1.0 / rms
        
        # Raw matmul output (before scaling)
        raw = x @ phi  # [batch, out_dim]
        
        # Split raw into components
        raw_pre = raw[:, :n]
        raw_post = raw[:, n:2*n]
        raw_res = raw[:, 2*n:]
        
        # Compute pre-activation values
        bias_pre = bias[:n]
        bias_post = bias[n:2*n]
        bias_res = bias[2*n:]
        
        scaled_pre = inv_rms * alpha_pre * raw_pre + bias_pre
        scaled_post = inv_rms * alpha_post * raw_post + bias_post
        scaled_res = inv_rms * alpha_res * raw_res + bias_res
        
        # ---- Backward through Sinkhorn-Knopp ----
        grad_scaled_res = sinkhorn_backward(
            scaled_res.reshape(batch, n, n),
            grad_H_res,
            ctx.sinkhorn_iters,
            ctx.eps,
        ).reshape(batch, n * n)
        
        # ---- Backward through H_post = 2 * sigmoid(scaled_post) ----
        sigmoid_post = torch.sigmoid(scaled_post)
        grad_scaled_post = grad_H_post * 2.0 * sigmoid_post * (1 - sigmoid_post)
        
        # ---- Backward through H_pre = normalize(sigmoid(scaled_pre)) ----
        sigmoid_pre = torch.sigmoid(scaled_pre)
        sigmoid_pre_sum = sigmoid_pre.sum(dim=-1, keepdim=True) + ctx.eps
        
        # d(H_pre_i)/d(sigmoid_pre_j) = delta_ij/sum - H_pre_i/sum
        grad_sigmoid_pre = (grad_H_pre - (grad_H_pre * H_pre).sum(dim=-1, keepdim=True)) / sigmoid_pre_sum
        
        # Through sigmoid
        grad_scaled_pre = grad_sigmoid_pre * sigmoid_pre * (1 - sigmoid_pre)
        
        # ---- Backward through scaling ----
        grad_raw_pre = grad_scaled_pre * inv_rms * alpha_pre
        grad_raw_post = grad_scaled_post * inv_rms * alpha_post
        grad_raw_res = grad_scaled_res * inv_rms * alpha_res
        
        grad_bias_pre = grad_scaled_pre.sum(dim=0)
        grad_bias_post = grad_scaled_post.sum(dim=0)
        grad_bias_res = grad_scaled_res.sum(dim=0)
        grad_bias = torch.cat([grad_bias_pre, grad_bias_post, grad_bias_res])
        
        grad_alpha_pre = (grad_scaled_pre * inv_rms * raw_pre).sum()
        grad_alpha_post = (grad_scaled_post * inv_rms * raw_post).sum()
        grad_alpha_res = (grad_scaled_res * inv_rms * raw_res).sum()
        
        # Gradient w.r.t. inv_rms
        grad_inv_rms = (
            (grad_scaled_pre * alpha_pre * raw_pre).sum(dim=-1, keepdim=True) +
            (grad_scaled_post * alpha_post * raw_post).sum(dim=-1, keepdim=True) +
            (grad_scaled_res * alpha_res * raw_res).sum(dim=-1, keepdim=True)
        )
        
        # inv_rms = 1 / rms, so d(inv_rms)/d(rms) = -1/rms^2 = -inv_rms^2
        grad_rms = grad_inv_rms * (-inv_rms * inv_rms)
        
        # rms = sqrt(sum(x^2) / in_dim + eps)
        # d(rms)/d(x) = x / (in_dim * rms)
        grad_x_from_rms = grad_rms * x / (in_dim * rms)
        
        # ---- Backward through matmul ----
        grad_raw = torch.cat([grad_raw_pre, grad_raw_post, grad_raw_res], dim=-1)
        grad_x_from_matmul = grad_raw @ phi.T
        grad_phi = x.T @ grad_raw
        
        # Combine gradients for x
        grad_x = grad_x_from_matmul + grad_x_from_rms
        
        # Convert gradients back to original dtypes
        return (
            grad_x.to(x_dtype),
            grad_phi.to(phi_dtype),
            grad_bias.to(bias_dtype),
            grad_alpha_pre.reshape(()).to(alpha_pre.dtype),
            grad_alpha_post.reshape(()).to(alpha_post.dtype),
            grad_alpha_res.reshape(()).to(alpha_res.dtype),
            None,  # sinkhorn_iters
            None,  # eps
        )


def sinkhorn_knopp(M: torch.Tensor, num_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """
    Project matrices to doubly stochastic via Sinkhorn-Knopp.

    Args:
        M: Input tensor (batch, 4, 4)
        num_iters: Number of row/column normalization iterations
        eps: Numerical stability constant

    Returns:
        Doubly stochastic tensor (batch, 4, 4) where all rows and columns sum to 1
    """
    return _SinkhornKnopp.apply(M, num_iters, eps)


def fused_stream_mix(
    H: torch.Tensor, H_pre: torch.Tensor, H_res: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused stream mixing operation.

    Args:
        H: Hyper-hidden state (batch, seq, 4, dim)
        H_pre: Pre-mixing weights (batch, 4), should sum to 1
        H_res: Residual mixing matrix (batch, 4, 4), doubly stochastic

    Returns:
        branch_input: Weighted sum of streams (batch, seq, dim)
        H_residual: Mixed residual streams (batch, seq, 4, dim)
    """
    return _FusedStreamMix.apply(H, H_pre, H_res)


def fused_add_residual(
    H_residual: torch.Tensor, branch_output: torch.Tensor, H_post: torch.Tensor,
) -> torch.Tensor:
    """
    Fused residual addition.

    Args:
        H_residual: Residual streams (batch, seq, 4, dim)
        branch_output: Layer output (batch, seq, dim)
        H_post: Post-distribution weights (batch, 4)

    Returns:
        H_new: Updated hyper-hidden (batch, seq, 4, dim)
    """
    return _FusedAddResidual.apply(H_residual, branch_output, H_post)


def fused_dynamic_weights(
    x: torch.Tensor,
    phi: torch.Tensor,
    bias: torch.Tensor,
    alpha_pre: torch.Tensor,
    alpha_post: torch.Tensor,
    alpha_res: torch.Tensor,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute dynamic connection weights with fused operations.
    
    Implements Eq. 14-19 from the DeepSeek mHC paper:
    1. Fused matmul: raw = x @ phi (Eq. 14)
    2. RMS norm: r = ||x||_2 / sqrt(in_dim) (Eq. 15)
    3. Scale + bias: scaled = (1/r) * alpha * raw + bias (Eq. 16)
    4. Activations: H_pre = norm(sigmoid(...)), H_post = 2*sigmoid(...) (Eq. 17-18)
    5. Sinkhorn-Knopp: H_res = sinkhorn(scaled_res) (Eq. 19) - FUSED INLINE
    
    Optimizations:
    - Transposed phi layout for coalesced memory access
    - Sinkhorn-Knopp fully fused inline (no separate kernel launch)
    - Single kernel computes Eq. 14-19 in one pass
    
    Args:
        x: Input tensor [batch, in_dim] - mean-pooled flattened hyper-hidden
        phi: Combined projection matrix [in_dim, n^2 + 2n]
        bias: Combined bias [n^2 + 2n] (absorbs base params)
        alpha_pre: Scale for H_pre projection (scalar tensor)
        alpha_post: Scale for H_post projection (scalar tensor)
        alpha_res: Scale for H_res projection (scalar tensor)
        sinkhorn_iters: Iterations for doubly stochastic projection
        eps: Numerical stability constant
        
    Returns:
        H_pre: Pre-mixing weights [batch, n], normalized to sum=1
        H_post: Post-distribution weights [batch, n], values in (0, 2)
        H_res: Residual mixing matrix [batch, n, n], doubly stochastic
    """
    return _FusedDynamicWeights.apply(
        x, phi, bias, alpha_pre, alpha_post, alpha_res, sinkhorn_iters, eps
    )
