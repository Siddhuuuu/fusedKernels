"""
MoE MLP layer using SORTED dispatch instead of masked dispatch.

bench_moe_layer_compare.py showed the router-level wins (v1/v2 fixes)
essentially evaporate at the full-layer level, especially at high expert
counts (128 experts: naive 575ms vs v2-routed masked-loop 912ms — SLOWER).
Root cause: FusedMoEMLP's expert loop uses boolean masking each iteration
(`match.any()`, advanced/boolean indexing) — that overhead scales with
expert count and swamps anything happening in the router.

This version fixes the loop itself, not just the router:
  1. Flatten [N, K] token-expert assignments into N*K (token, expert,
     weight) triples (a token used by K experts appears K times).
  2. Sort those triples by expert id — this groups all tokens assigned to
     the same expert into one CONTIGUOUS block.
  3. Gather token features into that sorted order once.
  4. Loop over experts, but each iteration is a cheap contiguous slice
     (no mask, no boolean indexing) — just plain start:end.
  5. Weight each expert's output by its combine weight, then scatter-add
     back to the original token positions (a token's final output is the
     sum of its K weighted expert outputs).

This is standard practice in real MoE systems short of a fully custom
fused dispatch kernel (which would need a hand-written grouped-GEMM
Triton kernel — a much bigger undertaking). It's still not a single
fused kernel launch — there's still a Python loop over experts — but the
per-iteration cost of that loop drops substantially since there's no
mask construction or advanced indexing left inside it.

Usage:
    from fusedkernels.moe_layer_sorted import FusedMoEMLPSorted
    moe = FusedMoEMLPSorted(dim=1024, hidden_dim=2048, n_experts=128, k=8).cuda()
    out = moe(x)
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
        x_flat = x.reshape(-1, self.dim)  # [N, dim]
        N = x_flat.shape[0]
        device = x_flat.device

        router_logits = self.router(x_flat)
        route_fn = fused_moe_route_v2 if self.router_version == "v2" else fused_moe_route
        topk_weights, topk_idx = route_fn(router_logits, self.k)  # [N, K] each

        # flatten (token, expert, weight) triples: N*K total assignments
        flat_expert_idx = topk_idx.reshape(-1)                                  # [N*K]
        flat_weight = topk_weights.reshape(-1)                                  # [N*K]
        flat_token_idx = (
            torch.arange(N, device=device).unsqueeze(1).expand(N, self.k).reshape(-1)
        )  # [N*K]

        # sort by expert id -> groups all tokens for a given expert into one
        # contiguous block, so each expert's slice is a plain range, not a mask
        sort_order = flat_expert_idx.argsort()
        sorted_expert_idx = flat_expert_idx[sort_order]
        sorted_token_idx = flat_token_idx[sort_order]
        sorted_weight = flat_weight[sort_order]

        gathered_x = x_flat[sorted_token_idx]  # [N*K, dim], grouped contiguously by expert

        # single host-device sync to get per-expert counts as plain python
        # ints (avoids doing .item() inside the loop, which would sync once
        # per expert instead of once total)
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
            chunk = gathered_x[start:start + cnt]        # plain contiguous slice, no mask
            out_sorted[start:start + cnt] = expert(chunk)

        out_sorted = out_sorted * sorted_weight.unsqueeze(-1).to(out_sorted.dtype)

        # scatter-add weighted expert outputs back to original token slots
        # (a token routed to K experts sums its K weighted contributions)
        out = torch.zeros_like(x_flat)
        out.index_add_(0, sorted_token_idx, out_sorted)

        return out.reshape(orig_shape)
