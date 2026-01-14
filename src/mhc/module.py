"""HyperConnection module for transformer architectures."""

import torch
import torch.nn as nn
from typing import Tuple, Callable

from .ops import sinkhorn_knopp, fused_stream_mix, fused_add_residual, fused_dynamic_weights


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
        use_fused_weights: If True, use fused kernel for dynamic weights (Eq. 14-19)

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
        eps: float = 1e-6,
        use_fused_weights: bool = True,
    ):
        super().__init__()

        if num_streams != 4:
            raise ValueError("Triton kernels require num_streams=4")

        self.dim = dim
        self.num_streams = num_streams
        self.dynamic = dynamic
        self.sinkhorn_iters = sinkhorn_iters
        self.layer_idx = layer_idx
        self.eps = eps
        self.use_fused_weights = use_fused_weights and dynamic

        n = num_streams
        in_dim = n * dim  # nC
        out_dim = n * n + 2 * n  # n^2 + 2n = 24 for n=4

        # Base parameters (used as bias in fused kernel)
        self.H_post_base = nn.Parameter(torch.ones(n))

        H_pre_init = torch.zeros(n)
        H_pre_init[layer_idx % n] = 1.0
        self.H_pre_base = nn.Parameter(H_pre_init)

        self.H_res_base = nn.Parameter(torch.eye(n))

        # Dynamic weight predictors
        if dynamic:
            if use_fused_weights:
                # Fused: single combined projection matrix phi [in_dim, out_dim]
                # Layout: [H_pre (n), H_post (n), H_res (n*n)]
                self.phi = nn.Parameter(torch.zeros(in_dim, out_dim))
                
                # Learnable scales (alpha_pre, alpha_post, alpha_res)
                self.alpha_pre = nn.Parameter(torch.tensor(init_scale))
                self.alpha_post = nn.Parameter(torch.tensor(init_scale))
                self.alpha_res = nn.Parameter(torch.tensor(init_scale))
            else:
                # Original: separate projection matrices
                self.W_post = nn.Linear(in_dim, n, bias=False)
                self.W_pre = nn.Linear(in_dim, n, bias=False)
                self.W_res = nn.Linear(in_dim, n * n, bias=False)

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

        if self.use_fused_weights:
            # Fused kernel path V2 (Eq. 14-19 from paper)
            # Optimizations: transposed phi, fully fused Sinkhorn, coalesced access
            # Mean pool across sequence: [batch, seq, n, dim] -> [batch, n*dim]
            x = H.mean(dim=1).reshape(batch, n * self.dim)
            
            # Build combined bias from base parameters
            # Layout: [H_pre_base (n), H_post_base (n), H_res_base (n*n)]
            bias = torch.cat([
                self.H_pre_base,
                self.H_post_base,
                self.H_res_base.flatten(),
            ])
            
            H_pre, H_post, H_res = fused_dynamic_weights(
                x, self.phi, bias,
                self.alpha_pre, self.alpha_post, self.alpha_res,
                self.sinkhorn_iters, self.eps,
            )
            
            return H_post.contiguous(), H_pre.contiguous(), H_res.contiguous()

        # Original path: separate projections
        # Mean pool with RMSNorm (separate norm layer)
        H_flat = H.mean(dim=1).reshape(batch, n * self.dim)
        # Apply RMSNorm manually since we don't have the norm layer in this path
        rms = torch.sqrt((H_flat * H_flat).mean(dim=-1, keepdim=True) + self.eps)
        H_flat = H_flat / rms

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
            f"dynamic={self.dynamic}, sinkhorn_iters={self.sinkhorn_iters}, "
            f"use_fused_weights={self.use_fused_weights}"
        )

