"""
mHC-Triton: Manifold-Constrained Hyper-Connections with fused Triton kernels.

High-performance implementation of multi-stream residual connections for transformers,
with doubly-stochastic mixing matrices projected via Sinkhorn-Knopp.

Reference: https://arxiv.org/html/2512.24880
"""

from .module import HyperConnection
from .ops import sinkhorn_knopp, fused_stream_mix, fused_add_residual
from ._torch_baseline import (
    sinkhorn_knopp_torch,
    fused_stream_mix_torch,
    fused_add_residual_torch,
    HyperConnectionTorch,
)

__version__ = "0.1.0"
__all__ = [
    "HyperConnection",
    "sinkhorn_knopp",
    "fused_stream_mix",
    "fused_add_residual",
    # PyTorch baselines for benchmarking
    "sinkhorn_knopp_torch",
    "fused_stream_mix_torch",
    "fused_add_residual_torch",
    "HyperConnectionTorch",
]

