"""
Sweep benchmark: native PyTorch vs v1 (sequential top-k) vs v2 (bitonic
sort top-k) MoE routing, across a grid of (n_experts, k) — this produces
the crossover-point data: where does fusion help, where does it hurt, and
does v2 push that boundary out.

Run:
    python examples/moe_router_sweep.py

Prints a table; also saves results to moe_sweep_results.csv for plotting.
"""

import csv
import time
import statistics
import torch
import torch.nn.functional as F

from fusedkernels.moe_routing import fused_moe_route
from fusedkernels.moe_routing_v2 import fused_moe_route_v2


def _bench(fn, *args, warmup=30, iters=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    time.sleep(0.05)  # let clocks/scheduling settle before timing
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms


def _bench_median(fn, *args, trials=3, **kwargs):
    times = [_bench(fn, *args, **kwargs) for _ in range(trials)]
    return statistics.median(times)


def naive_step(logits, k):
    logits = logits.clone().requires_grad_(True)
    probs = F.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    w.sum().backward()


def v1_step(logits, k):
    logits = logits.clone().requires_grad_(True)
    w, idx = fused_moe_route(logits, k)
    w.sum().backward()


def v2_step(logits, k):
    logits = logits.clone().requires_grad_(True)
    w, idx = fused_moe_route_v2(logits, k)
    w.sum().backward()


def main():
    assert torch.cuda.is_available(), "requires a CUDA GPU"
    N = 16384

    configs = [
        (8, 1), (8, 2), (8, 4),
        (16, 2), (16, 4), (16, 8),
        (32, 4), (32, 8), (32, 16),
        (64, 4), (64, 8), (64, 16), (64, 32),
        (128, 8), (128, 16), (128, 32),
    ]

    rows = []
    print(f"{'n_experts':>10}{'k':>6}{'native (ms)':>14}{'v1 (ms)':>12}{'v2 (ms)':>12}"
          f"{'v1 speedup':>13}{'v2 speedup':>13}")
    print("-" * 90)

    for n_experts, k in configs:
        logits = torch.randn(N, n_experts, device="cuda")
        t_native = _bench_median(naive_step, logits, k)
        t_v1 = _bench_median(v1_step, logits, k)
        t_v2 = _bench_median(v2_step, logits, k)

        v1_speedup = t_native / t_v1
        v2_speedup = t_native / t_v2

        print(f"{n_experts:>10}{k:>6}{t_native:>14.3f}{t_v1:>12.3f}{t_v2:>12.3f}"
              f"{v1_speedup:>12.2f}x{v2_speedup:>12.2f}x")

        rows.append({
            "n_experts": n_experts, "k": k,
            "native_ms": t_native, "v1_ms": t_v1, "v2_ms": t_v2,
            "v1_speedup": v1_speedup, "v2_speedup": v2_speedup,
        })

    with open("moe_sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved results to moe_sweep_results.csv")


if __name__ == "__main__":
    main()
