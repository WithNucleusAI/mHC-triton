"""High-level ops with autograd support for Hyper-Connections."""

import torch
from typing import Tuple

from ._kernels import sinkhorn_forward, stream_mix_forward, add_residual_forward
from ._backward import sinkhorn_backward, stream_mix_backward, add_residual_backward


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

