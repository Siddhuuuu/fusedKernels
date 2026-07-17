"""
Correctness tests for FusedMoEMLPSorted: forward output AND gradients must
match a naive reference MoE MLP exactly (same weights, same routing
decision), since the sort/gather/scatter restructuring is new and has
real bug risk (index bookkeeping, permutation correctness).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA GPU")


class NaiveMoEMLP(nn.Module):
    """Reference: masked loop + naive (non-fused) softmax/topk routing."""
    def __init__(self, dim, hidden_dim, n_experts, k, experts):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = experts  # share weights with the layer under test

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


@pytest.mark.parametrize("n_experts,k", [(8, 2), (32, 4), (64, 8), (128, 16)])
def test_sorted_dispatch_matches_naive(n_experts, k):
    from fusedkernels.moe_layer_sorted import FusedMoEMLPSorted

    torch.manual_seed(0)
    B, T, D, H = 2, 128, 256, 512

    sorted_layer = FusedMoEMLPSorted(D, H, n_experts, k, router_version="v2").cuda()
    naive_layer = NaiveMoEMLP(D, H, n_experts, k, sorted_layer.experts).cuda()
    naive_layer.router.load_state_dict(sorted_layer.router.state_dict())

    x_sorted = torch.randn(B, T, D, device="cuda", requires_grad=True)
    x_naive = x_sorted.detach().clone().requires_grad_(True)

    out_sorted = sorted_layer(x_sorted)
    out_naive = naive_layer(x_naive)

    assert torch.allclose(out_sorted, out_naive, atol=1e-3, rtol=1e-3), \
        f"sorted-dispatch output mismatch at n_experts={n_experts}, k={k}"

    grad_out = torch.randn_like(out_sorted)
    out_sorted.backward(grad_out)
    out_naive.backward(grad_out)

    assert torch.allclose(x_sorted.grad, x_naive.grad, atol=1e-3, rtol=1e-3), \
        f"sorted-dispatch input gradient mismatch at n_experts={n_experts}, k={k}"
    assert torch.allclose(
        sorted_layer.router.weight.grad, naive_layer.router.weight.grad, atol=1e-3, rtol=1e-3
    ), f"sorted-dispatch router gradient mismatch at n_experts={n_experts}, k={k}"


def test_sorted_dispatch_handles_empty_experts():
    """With very few tokens and many experts, some experts get zero tokens
    routed to them — make sure the `if cnt == 0: continue` path is correct
    and doesn't break gradients for unused experts."""
    from fusedkernels.moe_layer_sorted import FusedMoEMLPSorted

    torch.manual_seed(1)
    layer = FusedMoEMLPSorted(dim=64, hidden_dim=128, n_experts=64, k=2, router_version="v2").cuda()
    x = torch.randn(1, 4, 64, device="cuda", requires_grad=True)  # only 4 tokens, 64 experts

    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
