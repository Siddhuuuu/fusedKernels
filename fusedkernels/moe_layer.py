"""
A usable MoE MLP block built around the fused router (BASELINE dispatch
version — kept as the naive comparison point; see moe_layer_sorted.py for
the optimized version).

Scope note: this fuses the *router* but uses straightforward masked
matmuls for expert dispatch (loop over experts, boolean-mask tokens not
routed to that expert). Benchmarking showed this masked-loop dispatch —
not the router — becomes the dominant bottleneck at higher expert counts
(boolean masking + advanced indexing overhead scales with expert count).
moe_layer_sorted.py fixes this with sorted/contiguous dispatch instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_routing import fused_moe_route
from .moe_routing_v2 import fused_moe_route_v2


class Expert(nn.Module):
    """A single expert MLP (SwiGLU-style)."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FusedMoEMLP(nn.Module):
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

        router_logits = self.router(x_flat)
        route_fn = fused_moe_route_v2 if self.router_version == "v2" else fused_moe_route
        topk_weights, topk_idx = route_fn(router_logits, self.k)

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
