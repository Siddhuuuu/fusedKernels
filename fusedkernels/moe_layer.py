"""
A usable MoE MLP block built around the fused router.

Honest scope note: this fuses the *router* (softmax + top-k + renormalize)
into one kernel. The expert computation itself uses straightforward masked
matmuls (loop over experts, mask tokens not routed to that expert) rather
than a fully fused token-dispatch/permute/combine kernel. Building a fused
dispatch kernel (physically gathering each expert's tokens into a
contiguous buffer, computing, then scattering results back) is a
meaningfully bigger project — that's what makes libraries like Megablocks
non-trivial — and is the natural "next step" if you want to push further.

This layer is still a real, usable, drop-in MoE MLP for a transformer
block, and the fused router alone is fusing three separate kernel
launches (softmax, topk, renormalize) into one for every token at every
MoE layer, which is a real, currently-common bottleneck.

Usage:
    from fusedkernels.moe_layer import FusedMoEMLP
    moe = FusedMoEMLP(dim=1024, hidden_dim=2048, n_experts=8, k=2).cuda()
    out = moe(x)   # x: [B, T, dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_routing import fused_moe_route
from .moe_routing_v2 import fused_moe_route_v2


class Expert(nn.Module):
    """A single expert MLP (SwiGLU-style, matches typical MoE experts)."""
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
        x_flat = x.reshape(-1, self.dim)  # [N, dim]
        N = x_flat.shape[0]

        router_logits = self.router(x_flat)  # [N, n_experts]
        route_fn = fused_moe_route_v2 if self.router_version == "v2" else fused_moe_route
        topk_weights, topk_idx = route_fn(router_logits, self.k)  # [N, K] each

        out = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            # find (token, slot) pairs routed to this expert
            match = topk_idx == expert_id            # [N, K] bool
            token_mask = match.any(dim=-1)            # [N] bool: does this token use this expert
            if not token_mask.any():
                continue
            weight_for_expert = (topk_weights * match).sum(dim=-1)  # [N] combine weight (0 if not routed here)

            expert_out = expert(x_flat[token_mask])                 # [n_selected, dim]
            out[token_mask] += expert_out * weight_for_expert[token_mask].unsqueeze(-1)

        return out.reshape(orig_shape)
