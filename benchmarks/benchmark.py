#!/usr/bin/env python3
"""Benchmark script comparing Triton kernels vs PyTorch baselines.

Usage:
    python benchmarks/benchmark.py

This will output a markdown table suitable for pasting into README.md
"""

import torch
import time
from typing import Callable, Tuple
import argparse

from mhc import (
    sinkhorn_knopp,
    fused_stream_mix,
    fused_add_residual,
    sinkhorn_knopp_torch,
    fused_stream_mix_torch,
    fused_add_residual_torch,
    HyperConnection,
    HyperConnectionTorch,
)


def benchmark_fn(
    fn: Callable,
    args: Tuple,
    warmup: int = 10,
    repeats: int = 100,
    sync: bool = True,
) -> float:
    """Benchmark a function and return mean time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    if sync:
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    if sync:
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / repeats * 1000  # Convert to ms


def benchmark_sinkhorn(batch: int, num_iters: int = 20, device: str = "cuda") -> Tuple[float, float]:
    """Benchmark Sinkhorn-Knopp projection."""
    M = torch.randn(batch, 4, 4, device=device, dtype=torch.float16)

    # Triton
    triton_time = benchmark_fn(lambda m: sinkhorn_knopp(m, num_iters=num_iters), (M,))

    # PyTorch
    torch_time = benchmark_fn(lambda m: sinkhorn_knopp_torch(m, num_iters=num_iters), (M,))

    return torch_time, triton_time


def benchmark_stream_mix(
    batch: int, seq: int, dim: int, device: str = "cuda"
) -> Tuple[float, float]:
    """Benchmark stream mixing."""
    H = torch.randn(batch, seq, 4, dim, device=device, dtype=torch.float16)
    H_pre = torch.softmax(torch.randn(batch, 4, device=device, dtype=torch.float16), dim=-1)
    H_res = torch.randn(batch, 4, 4, device=device, dtype=torch.float16)
    # Make doubly stochastic for realistic test
    H_res = sinkhorn_knopp(H_res, num_iters=20)

    # Triton
    triton_time = benchmark_fn(lambda h, hp, hr: fused_stream_mix(h, hp, hr), (H, H_pre, H_res))

    # PyTorch
    torch_time = benchmark_fn(lambda h, hp, hr: fused_stream_mix_torch(h, hp, hr), (H, H_pre, H_res))

    return torch_time, triton_time


def benchmark_add_residual(
    batch: int, seq: int, dim: int, device: str = "cuda"
) -> Tuple[float, float]:
    """Benchmark residual addition."""
    H_residual = torch.randn(batch, seq, 4, dim, device=device, dtype=torch.float16)
    branch_output = torch.randn(batch, seq, dim, device=device, dtype=torch.float16)
    H_post = torch.softmax(torch.randn(batch, 4, device=device, dtype=torch.float16), dim=-1)

    # Triton
    triton_time = benchmark_fn(
        lambda hr, bo, hp: fused_add_residual(hr, bo, hp),
        (H_residual, branch_output, H_post),
    )

    # PyTorch
    torch_time = benchmark_fn(
        lambda hr, bo, hp: fused_add_residual_torch(hr, bo, hp),
        (H_residual, branch_output, H_post),
    )

    return torch_time, triton_time


def benchmark_full_forward_backward(
    batch: int, seq: int, dim: int, device: str = "cuda"
) -> Tuple[float, float]:
    """Benchmark full forward+backward through HyperConnection."""
    # Setup Triton module
    hc_triton = HyperConnection(dim=dim, num_streams=4, dynamic=True).to(device).half()

    # Setup PyTorch baseline module with same weights
    hc_torch = HyperConnectionTorch(dim=dim, num_streams=4, dynamic=True).to(device).half()
    hc_torch.load_state_dict(hc_triton.state_dict())

    H = torch.randn(batch, seq, 4, dim, device=device, dtype=torch.float16, requires_grad=True)

    def forward_backward_triton():
        H_in = H.detach().requires_grad_(True)
        branch_input, add_residual = hc_triton(H_in)
        # Simulate a simple layer
        branch_output = branch_input * 0.5
        H_new = add_residual(branch_output)
        loss = H_new.sum()
        loss.backward()
        return H_in.grad

    def forward_backward_torch():
        H_in = H.detach().requires_grad_(True)
        branch_input, add_residual = hc_torch(H_in)
        # Simulate a simple layer
        branch_output = branch_input * 0.5
        H_new = add_residual(branch_output)
        loss = H_new.sum()
        loss.backward()
        return H_in.grad

    # Triton version
    triton_time = benchmark_fn(forward_backward_triton, ())

    # PyTorch baseline
    torch_time = benchmark_fn(forward_backward_torch, ())

    return torch_time, triton_time


def measure_peak_memory(fn: Callable, warmup: int = 3) -> float:
    """Measure peak GPU memory usage of a function in MB."""
    # Warmup and clear
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 * 1024)  # Convert to MB


def benchmark_sinkhorn_memory(
    batch: int, num_iters: int = 20, device: str = "cuda"
) -> Tuple[float, float]:
    """Benchmark memory usage for Sinkhorn backward pass."""
    M = torch.randn(batch, 4, 4, device=device, dtype=torch.float16, requires_grad=True)

    def sinkhorn_backward_triton():
        M_in = M.detach().requires_grad_(True)
        P = sinkhorn_knopp(M_in, num_iters=num_iters)
        loss = P.sum()
        loss.backward()
        return M_in.grad

    def sinkhorn_backward_torch():
        M_in = M.detach().requires_grad_(True)
        P = sinkhorn_knopp_torch(M_in, num_iters=num_iters)
        loss = P.sum()
        loss.backward()
        return M_in.grad

    triton_mem = measure_peak_memory(sinkhorn_backward_triton)
    torch_mem = measure_peak_memory(sinkhorn_backward_torch)

    return torch_mem, triton_mem


def benchmark_full_memory(
    batch: int, seq: int, dim: int, device: str = "cuda"
) -> Tuple[float, float]:
    """Benchmark memory usage for full forward+backward."""
    # Setup Triton module
    hc_triton = HyperConnection(dim=dim, num_streams=4, dynamic=True).to(device).half()

    # Setup PyTorch baseline module with same weights
    hc_torch = HyperConnectionTorch(dim=dim, num_streams=4, dynamic=True).to(device).half()
    hc_torch.load_state_dict(hc_triton.state_dict())

    H = torch.randn(batch, seq, 4, dim, device=device, dtype=torch.float16, requires_grad=True)

    def forward_backward_triton():
        H_in = H.detach().requires_grad_(True)
        branch_input, add_residual = hc_triton(H_in)
        branch_output = branch_input * 0.5
        H_new = add_residual(branch_output)
        loss = H_new.sum()
        loss.backward()
        return H_in.grad

    def forward_backward_torch():
        H_in = H.detach().requires_grad_(True)
        branch_input, add_residual = hc_torch(H_in)
        branch_output = branch_input * 0.5
        H_new = add_residual(branch_output)
        loss = H_new.sum()
        loss.backward()
        return H_in.grad

    triton_mem = measure_peak_memory(forward_backward_triton)
    torch_mem = measure_peak_memory(forward_backward_torch)

    return torch_mem, triton_mem


def benchmark_simple_residual(
    batch: int, seq: int, dim: int, device: str = "cuda"
) -> float:
    """Benchmark simple residual connection forward+backward."""
    # Simple residual: x_out = x + layer(x)
    # Use same total tensor size as HyperConnection (batch, seq, 4, dim) flattened
    # to (batch, seq, 4*dim) for a fair comparison of data movement
    x = torch.randn(batch, seq, 4 * dim, device=device, dtype=torch.float16, requires_grad=True)

    def forward_backward_simple():
        x_in = x.detach().requires_grad_(True)
        # Simulate a simple layer (same as in HyperConnection benchmark)
        layer_output = x_in * 0.5
        # Simple residual addition
        x_out = x_in + layer_output
        loss = x_out.sum()
        loss.backward()
        return x_in.grad

    return benchmark_fn(forward_backward_simple, ())


def benchmark_simple_residual_memory(
    batch: int, seq: int, dim: int, device: str = "cuda"
) -> float:
    """Benchmark memory usage for simple residual connection."""
    x = torch.randn(batch, seq, 4 * dim, device=device, dtype=torch.float16, requires_grad=True)

    def forward_backward_simple():
        x_in = x.detach().requires_grad_(True)
        layer_output = x_in * 0.5
        x_out = x_in + layer_output
        loss = x_out.sum()
        loss.backward()
        return x_in.grad

    return measure_peak_memory(forward_backward_simple)


def main():
    parser = argparse.ArgumentParser(description="Benchmark mHC-Triton kernels")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seq", type=int, default=2048, help="Sequence length")
    parser.add_argument("--dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"mHC-Triton Benchmark")
    print(f"{'='*60}")
    print(f"Config: batch={args.batch}, seq={args.seq}, dim={args.dim}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"{'='*60}\n")

    results = []

    # Sinkhorn benchmark
    # Use batch * seq for Sinkhorn to match the scale of operations
    sinkhorn_batch = args.batch * args.seq
    torch_time, triton_time = benchmark_sinkhorn(sinkhorn_batch, num_iters=20)
    speedup = torch_time / triton_time
    results.append(("Sinkhorn (20 iter)", torch_time, triton_time, speedup))
    print(f"Sinkhorn (20 iter):      PyTorch {torch_time:.2f}ms | Triton {triton_time:.2f}ms | Speedup {speedup:.1f}x")

    # Stream Mix benchmark
    torch_time, triton_time = benchmark_stream_mix(args.batch, args.seq, args.dim)
    speedup = torch_time / triton_time
    results.append(("Stream Mix", torch_time, triton_time, speedup))
    print(f"Stream Mix:              PyTorch {torch_time:.2f}ms | Triton {triton_time:.2f}ms | Speedup {speedup:.1f}x")

    # Add Residual benchmark
    torch_time, triton_time = benchmark_add_residual(args.batch, args.seq, args.dim)
    speedup = torch_time / triton_time
    results.append(("Add Residual", torch_time, triton_time, speedup))
    print(f"Add Residual:            PyTorch {torch_time:.2f}ms | Triton {triton_time:.2f}ms | Speedup {speedup:.1f}x")

    # Full Forward+Backward benchmark
    torch_time, triton_time = benchmark_full_forward_backward(args.batch, args.seq, args.dim)
    speedup = torch_time / triton_time
    results.append(("Full Forward+Backward", torch_time, triton_time, speedup))
    print(f"Full Forward+Backward:   PyTorch {torch_time:.2f}ms | Triton {triton_time:.2f}ms | Speedup {speedup:.1f}x")

    # Memory benchmarks
    print(f"\n{'-'*60}")
    print("Memory Benchmarks:")
    print(f"{'-'*60}")

    memory_results = []

    # Sinkhorn memory
    torch_mem, triton_mem = benchmark_sinkhorn_memory(sinkhorn_batch, num_iters=20)
    savings = torch_mem / triton_mem if triton_mem > 0 else float('inf')
    memory_results.append(("Sinkhorn Backward", torch_mem, triton_mem, savings))
    print(f"Sinkhorn Backward:       PyTorch {torch_mem:.1f}MB | Triton {triton_mem:.1f}MB | Savings {savings:.1f}x")

    # Full forward+backward memory
    torch_mem, triton_mem = benchmark_full_memory(args.batch, args.seq, args.dim)
    savings = torch_mem / triton_mem if triton_mem > 0 else float('inf')
    memory_results.append(("Full Forward+Backward", torch_mem, triton_mem, savings))
    print(f"Full Forward+Backward:   PyTorch {torch_mem:.1f}MB | Triton {triton_mem:.1f}MB | Savings {savings:.1f}x")

    # Comparison with simple residual
    print(f"\n{'-'*60}")
    print("Comparison vs Simple Residual Connection:")
    print(f"{'-'*60}")

    simple_time = benchmark_simple_residual(args.batch, args.seq, args.dim)
    simple_mem = benchmark_simple_residual_memory(args.batch, args.seq, args.dim)

    # Get HyperConnection Triton numbers (already computed)
    hc_time = results[-1][2]  # Triton time from Full Forward+Backward
    hc_mem = memory_results[-1][2]  # Triton memory from Full Forward+Backward

    time_overhead = hc_time / simple_time
    mem_overhead = hc_mem / simple_mem

    print(f"Simple Residual:          {simple_time:.2f}ms | {simple_mem:.1f}MB")
    print(f"HyperConnection (Triton): {hc_time:.2f}ms | {hc_mem:.1f}MB")
    print(f"Overhead:                 {time_overhead:.1f}x slower | {mem_overhead:.1f}x more memory")

    # Print markdown table
    print(f"\n{'='*60}")
    print("Markdown tables for README.md:")
    print(f"{'='*60}\n")
    print(f"Benchmarks on {torch.cuda.get_device_name()} (batch={args.batch}, seq={args.seq}, dim={args.dim}):\n")

    print("### Speed\n")
    print("| Operation | PyTorch | Triton | Speedup |")
    print("|-----------|---------|--------|---------|")
    for name, pt_time, tr_time, speedup in results:
        print(f"| {name} | {pt_time:.2f}ms | {tr_time:.2f}ms | {speedup:.1f}x |")

    print("\n### Memory\n")
    print("| Operation | PyTorch | Triton | Savings |")
    print("|-----------|---------|--------|---------|")
    for name, pt_mem, tr_mem, savings in memory_results:
        print(f"| {name} | {pt_mem:.1f}MB | {tr_mem:.1f}MB | {savings:.1f}x |")

    print("\n### vs Simple Residual\n")
    print("| Method | Time | Memory |")
    print("|--------|------|--------|")
    print(f"| Simple Residual | {simple_time:.2f}ms | {simple_mem:.1f}MB |")
    print(f"| HyperConnection (Triton) | {hc_time:.2f}ms | {hc_mem:.1f}MB |")
    print(f"| **Overhead** | **{time_overhead:.1f}x** | **{mem_overhead:.1f}x** |")

    print()


if __name__ == "__main__":
    main()

