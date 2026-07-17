"""
3D sweep: does the fusion crossover point shift with N (number of tokens)?
Small N (e.g. inference-time single batch) vs large N (e.g. big pretraining
batch) may behave differently — small N means kernel launch overhead
dominates more; large N means the actual per-token work dominates more.

Run:
    python examples/moe_router_sweep_n.py
"""

import csv
import time
import statistics
import torch
import torch.nn.functional as F

from fusedkernels.moe_routing_v2 import fused_moe_route_v2


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


def v2_step(logits, k):
    logits = logits.clone().requires_grad_(True)
    w, idx = fused_moe_route_v2(logits, k)
    w.sum().backward()


def main():
    assert torch.cuda.is_available()

    n_values = [512, 2048, 8192, 32768, 131072]
    ek_configs = [(64, 8), (64, 32), (128, 32)]  # a small/medium/large-stress mix

    rows = []
    print(f"{'N':>10}{'n_experts':>10}{'k':>5}{'native (ms)':>13}{'v2 (ms)':>11}{'v2 speedup':>12}")
    print("-" * 65)

    for n_experts, k in ek_configs:
        for N in n_values:
            logits = torch.randn(N, n_experts, device="cuda")
            t_native = _bench_median(naive_step, logits, k)
            t_v2 = _bench_median(v2_step, logits, k)
            speedup = t_native / t_v2

            print(f"{N:>10}{n_experts:>10}{k:>5}{t_native:>13.3f}{t_v2:>11.3f}{speedup:>11.2f}x")
            rows.append({"N": N, "n_experts": n_experts, "k": k,
                          "native_ms": t_native, "v2_ms": t_v2, "v2_speedup": speedup})

    with open("moe_sweep_n_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved to moe_sweep_n_results.csv")


if __name__ == "__main__":
    main()
