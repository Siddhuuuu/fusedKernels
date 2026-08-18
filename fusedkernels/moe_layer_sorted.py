"""
MoE MLP layer using SORTED dispatch instead of masked dispatch.

bench_moe_layer_compare.py showed the masked-loop dispatch in moe_layer.py
becomes the dominant bottleneck at higher expert counts — boolean-mask
construction and advanced indexing overhead scale with expert count,
enough to make the "optimized-router" version SLOWER than plain PyTorch
at 128 experts (measured: 0.61-0.62x, i.e. the fused router's own gains
were entirely swamped by dispatch overhead).

Fix: sort tokens by assigned expert once (contiguous per-expert blocks),
then each expert's dispatch is a plain contiguous slice — no mask
construction, no boolean/advanced indexing inside the loop.

Measured: 1.06-1.32x speedup over the naive masked-dispatch layer across
most expert counts on an NVIDIA T4 (recovers and exceeds native PyTorch
performance where the masked version was previously losing).
"""

import torch
import torch.nn as nn

from .moe_layer import Expert
from .moe_routing import fused_moe_route
from .moe_routing_v2 import fused_moe_route_v2


class FusedMoEMLPSorted(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_experts: int, k: int = 2, router_version: str = "v2"):
        super().__init__()
        assert router_version in ("v1", "v2")
        self.dim = dim
        self.n_experts = n_experts
        self.k = k
        self.router_version = router_version
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        N = x_flat.shape[0]
        device = x_flat.device

        router_logits = self.router(x_flat)
        route_fn = fused_moe_route_v2 if self.router_version == "v2" else fused_moe_route
        topk_weights, topk_idx = route_fn(router_logits, self.k)

        # flatten (token, expert, weight) triples: N*K total assignments
        flat_expert_idx = topk_idx.reshape(-1)
        flat_weight = topk_weights.reshape(-1)
        flat_token_idx = (
            torch.arange(N, device=device).unsqueeze(1).expand(N, self.k).reshape(-1)
        )

        # sort by expert id -> groups all tokens for a given expert into
        # one contiguous block, so each expert's slice is a plain range
        sort_order = flat_expert_idx.argsort()
        sorted_expert_idx = flat_expert_idx[sort_order]
        sorted_token_idx = flat_token_idx[sort_order]
        sorted_weight = flat_weight[sort_order]

        gathered_x = x_flat[sorted_token_idx]

        counts = torch.bincount(sorted_expert_idx, minlength=self.n_experts)
        offsets = torch.cumsum(counts, dim=0)
        starts = (offsets - counts).tolist()
        counts_list = counts.tolist()

        out_sorted = torch.empty_like(gathered_x)
        for expert_id, expert in enumerate(self.experts):
            cnt = counts_list[expert_id]
            if cnt == 0:
                continue
            start = starts[expert_id]
            chunk = gathered_x[start:start + cnt]
            out_sorted[start:start + cnt] = expert(chunk)

        # cast weight to match dtype before multiplying — avoids unwanted
        # float32 promotion under bf16/fp16 training (router weights are
        # always float32 for precision)
        out_sorted = out_sorted * sorted_weight.unsqueeze(-1).to(out_sorted.dtype)

        out = torch.zeros_like(x_flat)
        out.index_add_(0, sorted_token_idx, out_sorted)

        return out.reshape(orig_shape)
