"""
Benchmark: native PyTorch vs v1 (sequential) vs v2 (tl.sort based) vs v4
(experimental truncated bitonic) MoE routers, across a grid of
(n_experts, k) — the headline comparison for this whole project.

Run:
    python examples/moe_router_sweep.py
"""

import csv
import time
import statistics
import torch
import torch.nn.functional as F

from fusedkernels.moe_routing import fused_moe_route
from fusedkernels.moe_routing_v2 import fused_moe_route_v2
from fusedkernels.moe_routing_v4 import fused_moe_route_v4


def _bench(fn, *args, warmup=30, iters=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    time.sleep(0.05)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def _bench_median(fn, *args, trials=3, **kwargs):
    return statistics.median(_bench(fn, *args, **kwargs) for _ in range(trials))


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


def v4_step(logits, k):
    logits = logits.clone().requires_grad_(True)
    w, idx = fused_moe_route_v4(logits, k)
    w.sum().backward()


def main():
    assert torch.cuda.is_available()
    N = 16384

    configs = [
        (8, 1), (8, 2), (8, 4),
        (16, 2), (16, 4), (16, 8),
        (32, 4), (32, 8), (32, 16),
        (64, 4), (64, 8), (64, 16), (64, 32),
        (128, 8), (128, 16), (128, 32),
    ]

    rows = []
    print(f"{'n_experts':>10}{'k':>6}{'native':>10}{'v1':>10}{'v2':>10}{'v4':>10}"
          f"{'v1_spd':>9}{'v2_spd':>9}{'v4_spd':>9}")
    print("-" * 89)

    for n_experts, k in configs:
        logits = torch.randn(N, n_experts, device="cuda")
        t_native = _bench_median(naive_step, logits, k)
        t_v1 = _bench_median(v1_step, logits, k)
        t_v2 = _bench_median(v2_step, logits, k)
        t_v4 = _bench_median(v4_step, logits, k)

        v1s, v2s, v4s = t_native / t_v1, t_native / t_v2, t_native / t_v4

        print(f"{n_experts:>10}{k:>6}{t_native:>10.3f}{t_v1:>10.3f}{t_v2:>10.3f}{t_v4:>10.3f}"
              f"{v1s:>8.2f}x{v2s:>8.2f}x{v4s:>8.2f}x")

        rows.append({
            "n_experts": n_experts, "k": k,
            "native_ms": t_native, "v1_ms": t_v1, "v2_ms": t_v2, "v4_ms": t_v4,
            "v1_speedup": v1s, "v2_speedup": v2s, "v4_speedup": v4s,
        })

    with open("moe_sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved to moe_sweep_results.csv")


if __name__ == "__main__":
    main()
