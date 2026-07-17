"""
End-to-end comparison: a full MoE MLP block (router + experts) using the
fused router vs a naive-routing version, on realistic batch/seq shapes.

Run:
    python examples/bench_moe_layer.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusedkernels.moe_layer import FusedMoEMLP, Expert
from fusedkernels.moe_routing import fused_moe_route


class NaiveMoEMLP(nn.Module):
    """Same architecture as FusedMoEMLP but with naive softmax+topk+renormalize routing."""
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


def bench_layer(layer, x, warmup=10, iters=30):
    for _ in range(warmup):
        y = layer(x)
        y.sum().backward()
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        y = layer(x)
        y.sum().backward()
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed * 1000, peak_mem_gb


def main():
    assert torch.cuda.is_available(), "requires a CUDA GPU"
    B, T, D, H = 4, 512, 1024, 2048
    N_EXPERTS, K = 8, 2

    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)

    naive = NaiveMoEMLP(D, H, N_EXPERTS, K).cuda().to(torch.bfloat16)
    fused = FusedMoEMLP(D, H, N_EXPERTS, K).cuda().to(torch.bfloat16)
    fused.load_state_dict(naive.state_dict())  # same weights, fair comparison

    print(f"MoE MLP: dim={D} hidden={H} n_experts={N_EXPERTS} k={K} "
          f"batch={B} seq_len={T} dtype=bf16\n")

    t_naive, m_naive = bench_layer(naive, x)
    t_fused, m_fused = bench_layer(fused, x)

    print(f"naive router:  {t_naive:7.2f} ms/iter   peak mem: {m_naive:6.3f} GB")
    print(f"fused router:  {t_fused:7.2f} ms/iter   peak mem: {m_fused:6.3f} GB")
    print(f"speedup: {t_naive / t_fused:.2f}x   mem reduction: {(1 - m_fused / m_naive) * 100:.1f}%")


if __name__ == "__main__":
    main()
