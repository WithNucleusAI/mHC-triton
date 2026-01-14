# mHC-Triton

**Manifold-Constrained Hyper-Connections** with fused Triton kernels for efficient training.

Created by [NucleusAI](https://github.com/NucleusAI) • Authors: Chandan Akiti & Sai Ajay Modukuri

Based on the paper: [Hyper-Connections](https://arxiv.org/html/2512.24880) by the DeepSeek team

## Features

- **Fused Triton kernels** optimized for NVIDIA H100
- **4x4 stream mixing matrices** kept in registers
- **20x memory reduction** via backward recomputation
- **Full autograd support** for training

## Installation

```bash
pip install git+https://github.com/NucleusAI/mHC-triton.git
```

Or install from source:

```bash
git clone https://github.com/NucleusAI/mHC-triton.git
cd mHC-triton
pip install -e .
```

## Quick Start

```python
import torch
from mhc import HyperConnection

# Create hyper-connection layer
hc = HyperConnection(dim=512, num_streams=4, dynamic=True).cuda()

# Input: hyper-hidden state (batch, seq, num_streams, dim)
H = torch.randn(2, 128, 4, 512, device='cuda')

# Forward pass
branch_input, add_residual = hc(H)

# Your layer (e.g., attention, MLP)
branch_output = your_layer(branch_input)

# Combine with residual streams
H_new = add_residual(branch_output)
```

## Architecture

The hyper-connection module provides:

1. **Pre-mixing**: Combines streams into layer input via learned weights
2. **Residual mixing**: Transforms streams via doubly-stochastic matrix (Sinkhorn projection)
3. **Post-distribution**: Routes layer output back to streams

```
H (batch, seq, 4, dim)
       │
       ├──► Pre-mix ──► branch_input (batch, seq, dim)
       │                      │
       │                      ▼
       │                 Your Layer
       │                      │
       │                      ▼
       │                branch_output
       │                      │
       └──► Res-mix ──────────┴──► Add ──► H_new (batch, seq, 4, dim)
```

## API Reference

### `HyperConnection`

```python
HyperConnection(
    dim: int,              # Hidden dimension
    num_streams: int = 4,  # Number of parallel streams (must be 4)
    layer_idx: int = 0,    # Layer index for initialization
    dynamic: bool = True,  # Use input-dependent weights
    sinkhorn_iters: int = 20,  # Iterations for doubly-stochastic projection
    init_scale: float = 0.1,   # Initial scale for dynamic weight deltas
)
```

### Low-Level Ops

```python
from mhc import sinkhorn_knopp, fused_stream_mix, fused_add_residual

# Project to doubly-stochastic matrix
P = sinkhorn_knopp(M, num_iters=20)  # (batch, 4, 4) → (batch, 4, 4)

# Fused stream mixing
branch_input, H_residual = fused_stream_mix(H, H_pre, H_res)

# Fused residual addition
H_new = fused_add_residual(H_residual, branch_output, H_post)
```

## Performance

Benchmarks on H100 SXM5 (batch=8, seq=2048, dim=4096):

| Operation | PyTorch | Triton | Speedup |
|-----------|---------|--------|---------|
| Sinkhorn (20 iter) | 0.42ms | 0.08ms | 5.2x |
| Stream Mix | 1.85ms | 0.32ms | 5.8x |
| Full Forward+Backward | 8.2ms | 1.9ms | 4.3x |

Memory savings from recomputation: **20x** for Sinkhorn backward.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Triton ≥ 2.1
- CUDA GPU (optimized for H100)

## License

MIT License

