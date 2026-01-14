"""HyperConnection module for transformer architectures."""

import torch
import torch.nn as nn
from typing import Tuple, Callable

from .ops import sinkhorn_knopp, fused_stream_mix, fused_add_residual


class HyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connections using fused Triton kernels.

    Provides multi-stream residual connections with learned mixing weights,
    where the residual mixing matrix is constrained to be doubly stochastic
    via Sinkhorn-Knopp projection.

    Reference: https://arxiv.org/html/2512.24880

    Args:
        dim: Hidden dimension of each stream
        num_streams: Number of parallel streams (must be 4)
        layer_idx: Layer index, used for initialization
        dynamic: If True, compute input-dependent weights
        sinkhorn_iters: Iterations for doubly stochastic projection
        init_scale: Initial scale for dynamic weight deltas

    Example:
        >>> hc = HyperConnection(dim=512, num_streams=4).cuda()
        >>> H = torch.randn(2, 128, 4, 512, device='cuda')
        >>> branch_input, add_residual = hc(H)
        >>> branch_output = my_layer(branch_input)
        >>> H_new = add_residual(branch_output)
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
            raise ValueError("Triton kernels require num_streams=4")

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
            H_res = sinkhorn_knopp(
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

        # Project to doubly stochastic
        H_res = sinkhorn_knopp(
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
        branch_input, H_residual = fused_stream_mix(H, H_pre, H_res)

        def add_residual(branch_output: torch.Tensor) -> torch.Tensor:
            return fused_add_residual(H_residual, branch_output, H_post)

        return branch_input, add_residual

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_streams={self.num_streams}, "
            f"dynamic={self.dynamic}, sinkhorn_iters={self.sinkhorn_iters}"
        )

