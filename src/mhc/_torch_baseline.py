"""Pure PyTorch baseline implementations for benchmarking.

These are reference implementations of the Triton kernels for
performance comparison. They are numerically equivalent but slower.
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable


def sinkhorn_knopp_torch(
    M: torch.Tensor, num_iters: int = 20, eps: float = 1e-8
) -> torch.Tensor:
    """
    Project matrices to doubly stochastic via Sinkhorn-Knopp (PyTorch baseline).

    Args:
        M: Input tensor (batch, 4, 4)
        num_iters: Number of row/column normalization iterations
        eps: Numerical stability constant

    Returns:
        Doubly stochastic tensor (batch, 4, 4) where all rows and columns sum to 1
    """
    # Apply abs() + eps like the Triton kernel
    P = M.abs() + eps

    for _ in range(num_iters):
        # Row normalization
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        # Column normalization
        P = P / (P.sum(dim=-2, keepdim=True) + eps)

    return P


def fused_stream_mix_torch(
    H: torch.Tensor, H_pre: torch.Tensor, H_res: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stream mixing operation (PyTorch baseline).

    Computes:
    - branch_input = einsum('bn,bsnd->bsd', H_pre, H)
    - H_residual = einsum('bnm,bsmd->bsnd', H_res, H)

    Args:
        H: Hyper-hidden state (batch, seq, 4, dim)
        H_pre: Pre-mixing weights (batch, 4), should sum to 1
        H_res: Residual mixing matrix (batch, 4, 4), doubly stochastic

    Returns:
        branch_input: Weighted sum of streams (batch, seq, dim)
        H_residual: Mixed residual streams (batch, seq, 4, dim)
    """
    # branch_input = weighted sum of streams over stream dimension
    # H_pre: (batch, 4), H: (batch, seq, 4, dim)
    # einsum: 'bn,bsnd->bsd'
    branch_input = torch.einsum('bn,bsnd->bsd', H_pre, H)

    # H_residual = matrix multiply over stream dimension
    # H_res: (batch, 4, 4), H: (batch, seq, 4, dim)
    # einsum: 'bnm,bsmd->bsnd'
    H_residual = torch.einsum('bnm,bsmd->bsnd', H_res, H)

    return branch_input, H_residual


def fused_add_residual_torch(
    H_residual: torch.Tensor, branch_output: torch.Tensor, H_post: torch.Tensor
) -> torch.Tensor:
    """
    Residual addition operation (PyTorch baseline).

    Computes: H_new[n] = H_residual[n] + H_post[n] * branch_output

    Args:
        H_residual: Residual streams (batch, seq, 4, dim)
        branch_output: Layer output (batch, seq, dim)
        H_post: Post-distribution weights (batch, 4)

    Returns:
        H_new: Updated hyper-hidden (batch, seq, 4, dim)
    """
    # H_post: (batch, 4) -> (batch, 1, 4, 1) for broadcasting
    # branch_output: (batch, seq, dim) -> (batch, seq, 1, dim)
    H_new = H_residual + H_post[:, None, :, None] * branch_output[:, :, None, :]

    return H_new


class HyperConnectionTorch(nn.Module):
    """
    Pure PyTorch baseline of HyperConnection for benchmarking.

    This is functionally equivalent to HyperConnection but uses
    standard PyTorch ops instead of fused Triton kernels.

    Args:
        dim: Hidden dimension of each stream
        num_streams: Number of parallel streams (must be 4)
        layer_idx: Layer index, used for initialization
        dynamic: If True, compute input-dependent weights
        sinkhorn_iters: Iterations for doubly stochastic projection
        init_scale: Initial scale for dynamic weight deltas
    """

    def __init__(
        self,
        dim: int,
        num_streams: int = 4,
        layer_idx: int = 0,
        dynamic: bool = True,
        sinkhorn_iters: int = 20,
        init_scale: float = 0.1,
    ):
        super().__init__()

        if num_streams != 4:
            raise ValueError("num_streams must be 4 for compatibility")

        self.dim = dim
        self.num_streams = num_streams
        self.dynamic = dynamic
        self.sinkhorn_iters = sinkhorn_iters
        self.layer_idx = layer_idx

        n = num_streams

        # Base parameters
        self.H_post_base = nn.Parameter(torch.ones(n))

        H_pre_init = torch.zeros(n)
        H_pre_init[layer_idx % n] = 1.0
        self.H_pre_base = nn.Parameter(H_pre_init)

        self.H_res_base = nn.Parameter(torch.eye(n))

        # Dynamic weight predictors
        if dynamic:
            self.norm = nn.LayerNorm(dim)
            self.W_post = nn.Linear(n * dim, n, bias=False)
            self.W_pre = nn.Linear(n * dim, n, bias=False)
            self.W_res = nn.Linear(n * dim, n * n, bias=False)

            # Zero init for residual-like behavior at start
            nn.init.zeros_(self.W_post.weight)
            nn.init.zeros_(self.W_pre.weight)
            nn.init.zeros_(self.W_res.weight)

            self.scale_post = nn.Parameter(torch.tensor(init_scale))
            self.scale_pre = nn.Parameter(torch.tensor(init_scale))
            self.scale_res = nn.Parameter(torch.tensor(init_scale))

    def _compute_weights(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute constrained connection weights from input."""
        n = self.num_streams
        batch = H.shape[0]

        if not self.dynamic:
            # Static weights
            H_post = torch.sigmoid(self.H_post_base).unsqueeze(0).expand(batch, -1)
            H_pre = torch.sigmoid(self.H_pre_base)
            H_pre = (H_pre / (H_pre.sum() + 1e-8)).unsqueeze(0).expand(batch, -1)
            H_res = sinkhorn_knopp_torch(
                self.H_res_base.unsqueeze(0).expand(batch, -1, -1).contiguous(),
                self.sinkhorn_iters
            )
            return H_post.contiguous(), H_pre.contiguous(), H_res.contiguous()

        # Dynamic weights based on input
        H_norm = self.norm(H)
        H_flat = H_norm.reshape(batch, -1, n * self.dim).mean(dim=1)

        delta_post = self.W_post(H_flat) * self.scale_post
        delta_pre = self.W_pre(H_flat) * self.scale_pre
        delta_res = self.W_res(H_flat).reshape(batch, n, n) * self.scale_res

        # Apply constraints
        H_post = torch.sigmoid(self.H_post_base + delta_post)
        H_pre = torch.sigmoid(self.H_pre_base + delta_pre)
        H_pre = H_pre / (H_pre.sum(dim=-1, keepdim=True) + 1e-8)

        # Project to doubly stochastic (using PyTorch baseline)
        H_res = sinkhorn_knopp_torch(
            (self.H_res_base + delta_res).contiguous(),
            self.sinkhorn_iters
        )

        return H_post.contiguous(), H_pre.contiguous(), H_res.contiguous()

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """
        Forward pass through hyper-connection.

        Args:
            H: Hyper-hidden state (batch, seq, 4, dim)

        Returns:
            branch_input: Input for the layer (batch, seq, dim)
            add_residual: Callable to combine layer output with residual
        """
        H_post, H_pre, H_res = self._compute_weights(H)
        branch_input, H_residual = fused_stream_mix_torch(H, H_pre, H_res)

        def add_residual(branch_output: torch.Tensor) -> torch.Tensor:
            return fused_add_residual_torch(H_residual, branch_output, H_post)

        return branch_input, add_residual

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_streams={self.num_streams}, "
            f"dynamic={self.dynamic}, sinkhorn_iters={self.sinkhorn_iters}"
        )

