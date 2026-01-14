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


def fused_dynamic_weights_torch(
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
    Compute dynamic connection weights (PyTorch baseline).
    
    Implements Eq. 14-19 from the paper:
    1. Fused matmul: raw = x @ phi (Eq. 14)
    2. RMS norm: r = ||x||_2 / sqrt(in_dim) (Eq. 15)
    3. Scale + bias: scaled = (1/r) * alpha * raw + bias (Eq. 16)
    4. Activations: H_pre = norm(sigmoid(...)), H_post = 2*sigmoid(...) (Eq. 17-18)
    5. Sinkhorn-Knopp: H_res = sinkhorn(scaled_res) (Eq. 19)
    
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
    batch, in_dim = x.shape
    n = 4  # num_streams
    
    # Eq. 14: Compute all projections in one matmul
    raw = x @ phi  # [batch, n^2 + 2n]
    
    # Eq. 15: Compute RMS norm factor
    rms = torch.sqrt((x * x).sum(dim=-1, keepdim=True) / in_dim + eps)
    inv_rms = 1.0 / rms
    
    # Split into components
    raw_pre = raw[:, :n]
    raw_post = raw[:, n:2*n]
    raw_res = raw[:, 2*n:]
    
    bias_pre = bias[:n]
    bias_post = bias[n:2*n]
    bias_res = bias[2*n:]
    
    # Eq. 16: Scale + bias
    scaled_pre = inv_rms * alpha_pre * raw_pre + bias_pre
    scaled_post = inv_rms * alpha_post * raw_post + bias_post
    scaled_res = inv_rms * alpha_res * raw_res + bias_res
    
    # Eq. 17: H_pre = sigmoid then normalize to sum=1
    H_pre = torch.sigmoid(scaled_pre)
    H_pre = H_pre / (H_pre.sum(dim=-1, keepdim=True) + eps)
    
    # Eq. 18: H_post = 2 * sigmoid
    H_post = 2.0 * torch.sigmoid(scaled_post)
    
    # Eq. 19: Sinkhorn-Knopp on H_res
    H_res = sinkhorn_knopp_torch(
        scaled_res.reshape(batch, n, n),
        sinkhorn_iters,
        eps,
    )
    
    return H_pre, H_post, H_res


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
        use_fused_weights: If True, use fused weight computation (Eq. 14-19)
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
            raise ValueError("num_streams must be 4 for compatibility")

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
                # NOTE: keep these as 1D tensors with numel==1 for FSDP/fully_shard compatibility
                self.alpha_pre = nn.Parameter(torch.tensor([init_scale]))
                self.alpha_post = nn.Parameter(torch.tensor([init_scale]))
                self.alpha_res = nn.Parameter(torch.tensor([init_scale]))
            else:
                # Original: separate projection matrices
                self.W_post = nn.Linear(in_dim, n, bias=False)
                self.W_pre = nn.Linear(in_dim, n, bias=False)
                self.W_res = nn.Linear(in_dim, n * n, bias=False)

                # Zero init for residual-like behavior at start
                nn.init.zeros_(self.W_post.weight)
                nn.init.zeros_(self.W_pre.weight)
                nn.init.zeros_(self.W_res.weight)

                # NOTE: keep these as 1D tensors with numel==1 for FSDP/fully_shard compatibility
                self.scale_post = nn.Parameter(torch.tensor([init_scale]))
                self.scale_pre = nn.Parameter(torch.tensor([init_scale]))
                self.scale_res = nn.Parameter(torch.tensor([init_scale]))

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

        if self.use_fused_weights:
            # Fused path (Eq. 14-19 from paper)
            # Mean pool across sequence: [batch, seq, n, dim] -> [batch, n*dim]
            x = H.mean(dim=1).reshape(batch, n * self.dim)
            
            # Build combined bias from base parameters
            # Layout: [H_pre_base (n), H_post_base (n), H_res_base (n*n)]
            bias = torch.cat([
                self.H_pre_base,
                self.H_post_base,
                self.H_res_base.flatten(),
            ])
            
            H_pre, H_post, H_res = fused_dynamic_weights_torch(
                x, self.phi, bias,
                self.alpha_pre, self.alpha_post, self.alpha_res,
                self.sinkhorn_iters, self.eps,
            )
            
            return H_post.contiguous(), H_pre.contiguous(), H_res.contiguous()

        # Original path: separate projections
        # Mean pool with RMSNorm
        H_flat = H.mean(dim=1).reshape(batch, n * self.dim)
        # Apply RMSNorm manually
        rms = torch.sqrt((H_flat * H_flat).mean(dim=-1, keepdim=True) + self.eps)
        H_flat = H_flat / rms

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
            f"dynamic={self.dynamic}, sinkhorn_iters={self.sinkhorn_iters}, "
            f"use_fused_weights={self.use_fused_weights}"
        )

