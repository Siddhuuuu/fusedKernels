"""
Full MoE MLP layer comparison: naive dispatch vs v1/v2-routed masked
dispatch vs sorted dispatch. This is what revealed that dispatch overhead
(not the router) was the real bottleneck at higher expert counts.

Run:
    python examples/bench_moe_layer_compare.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusedkernels.moe_layer import FusedMoEMLP, Expert
from fusedkernels.moe_layer_sorted import FusedMoEMLPSorted


class NaiveMoEMLP(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts, k):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(n_experts)])

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        router_logits = self.router(x_flat)
        probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_idx = torch.topk(probs, self.k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            match = topk_idx == expert_id
            token_mask = match.any(dim=-1)
            if not token_mask.any():
                continue
            weight_for_expert = (topk_weights * match).sum(dim=-1)
            expert_out = expert(x_flat[token_mask])
            out[token_mask] += expert_out * weight_for_expert[token_mask].unsqueeze(-1)

        return out.reshape(orig_shape)


def bench_layer(layer, x, warmup=20, iters=100):
    for _ in range(warmup):
        y = layer(x)
        y.sum().backward()
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    time.sleep(0.05)

    start = time.perf_counter()
    for _ in range(iters):
        y = layer(x)
        y.sum().backward()
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def run_config(B, T, D, H, n_experts, k):
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)

    naive = NaiveMoEMLP(D, H, n_experts, k).cuda().to(torch.bfloat16)
    v2_masked = FusedMoEMLP(D, H, n_experts, k, router_version="v2").cuda().to(torch.bfloat16)
    v2_sorted = FusedMoEMLPSorted(D, H, n_experts, k, router_version="v2").cuda().to(torch.bfloat16)
    v2_masked.load_state_dict(naive.state_dict())
    v2_sorted.load_state_dict(naive.state_dict())

    t_naive = bench_layer(naive, x)
    t_masked = bench_layer(v2_masked, x)
    t_sorted = bench_layer(v2_sorted, x)

    print(f"experts={n_experts:>4} k={k:>3}  "
          f"naive={t_naive:8.3f}ms  v2_masked={t_masked:8.3f}ms  v2_sorted={t_sorted:8.3f}ms  "
          f"masked_spd={t_naive/t_masked:5.2f}x  sorted_spd={t_naive/t_sorted:5.2f}x")


def main():
    assert torch.cuda.is_available()
    B, T, D, H = 4, 512, 1024, 2048

    print(f"MoE MLP layer, batch={B} seq_len={T} dim={D} hidden={H} dtype=bf16\n")

    configs = [
        (8, 2), (16, 4), (32, 8), (64, 8), (64, 16), (128, 8), (128, 16), (128, 32),
    ]
    for n_experts, k in configs:
        run_config(B, T, D, H, n_experts, k)


if __name__ == "__main__":
    main()
