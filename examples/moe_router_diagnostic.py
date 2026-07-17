"""
Diagnostic: where is v2's time actually going at large K?

Hypothesis: the bitonic sort itself is O(log^2 E) and fast, but the
Python-level `for kk in range(K): extract element kk` readout loop after
the sort is still O(K) sequential steps — so we may have just moved the
bottleneck from "K sequential comparisons" to "K sequential extractions"
without actually removing the K-dependence.

This script times:
  1. Forward pass only (includes sort + extraction + normalize)
  2. Full forward + backward
  3. Native PyTorch topk forward, for reference

across increasing K at a fixed large expert count, to see whether v2's
forward time itself scales with K (confirming the extraction-loop
hypothesis) or stays flat (meaning the bottleneck is elsewhere, e.g. the
backward pass or kernel launch overhead).

Run:
    python examples/moe_router_diagnostic.py
"""

import time
import torch
import torch.nn.functional as F

from fusedkernels.moe_routing_v2 import fused_moe_route_v2


def _time_fn(fn, *args, warmup=30, iters=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    time.sleep(0.05)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def v2_fwd_only(logits, k):
    with torch.no_grad():
        fused_moe_route_v2(logits, k)


def v2_fwd_bwd(logits, k):
    logits = logits.clone().requires_grad_(True)
    w, idx = fused_moe_route_v2(logits, k)
    w.sum().backward()


def native_fwd_only(logits, k):
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        torch.topk(probs, k, dim=-1)


def main():
    assert torch.cuda.is_available()
    N = 16384
    n_experts = 128
    ks = [1, 2, 4, 8, 16, 32, 64]

    print(f"n_experts={n_experts}, N={N} tokens\n")
    print(f"{'K':>4}{'native fwd (ms)':>18}{'v2 fwd-only (ms)':>18}{'v2 fwd+bwd (ms)':>18}")
    print("-" * 60)

    for k in ks:
        logits = torch.randn(N, n_experts, device="cuda")

        t_native_fwd = _time_fn(native_fwd_only, logits, k)
        t_v2_fwd = _time_fn(v2_fwd_only, logits, k)
        t_v2_fwdbwd = _time_fn(v2_fwd_bwd, logits, k)

        print(f"{k:>4}{t_native_fwd:>18.4f}{t_v2_fwd:>18.4f}{t_v2_fwdbwd:>18.4f}")

    print("\nIf 'v2 fwd-only' grows roughly linearly with K, the extraction")
    print("loop (not the sort) is the bottleneck — confirms the hypothesis.")
    print("If 'v2 fwd-only' stays flat but 'v2 fwd+bwd' grows with K, the")
    print("backward pass's own K-loop (in the bwd kernel) is the culprit instead.")


if __name__ == "__main__":
    main()
