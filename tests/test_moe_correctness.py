"""Correctness tests for v1 (baseline) router and the naive MoE MLP layer."""

import pytest
import torch
import torch.nn.functional as F

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA GPU")


def naive_route(logits, k):
    probs = F.softmax(logits, dim=-1)
    topk_weights, topk_idx = torch.topk(probs, k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_idx


@pytest.mark.parametrize("n_experts,k", [(8, 2), (16, 4), (64, 8), (128, 1)])
def test_fused_moe_route_matches_naive(n_experts, k):
    from fusedkernels.moe_routing import fused_moe_route

    torch.manual_seed(0)
    N = 1024
    logits_ref = torch.randn(N, n_experts, device="cuda", requires_grad=True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)

    w_ref, idx_ref = naive_route(logits_ref, k)
    w_fused, idx_fused = fused_moe_route(logits_fused, k)

    ref_sets = [set(row.tolist()) for row in idx_ref]
    fused_sets = [set(row.tolist()) for row in idx_fused]
    assert ref_sets == fused_sets

    def sorted_weights(idx, w):
        order = idx.argsort(dim=-1)
        return torch.gather(w, -1, order)

    assert torch.allclose(sorted_weights(idx_ref, w_ref), sorted_weights(idx_fused, w_fused),
                           atol=1e-3, rtol=1e-3)


def test_fused_moe_route_weights_sum_to_one():
    from fusedkernels.moe_routing import fused_moe_route

    torch.manual_seed(1)
    logits = torch.randn(512, 32, device="cuda")
    w, _ = fused_moe_route(logits, k=4)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_fused_moe_mlp_runs_and_shapes_match():
    from fusedkernels.moe_layer import FusedMoEMLP

    torch.manual_seed(2)
    B, T, D = 2, 64, 256
    moe = FusedMoEMLP(dim=D, hidden_dim=512, n_experts=8, k=2, router_version="v1").cuda()
    x = torch.randn(B, T, D, device="cuda", requires_grad=True)

    out = moe(x)
    assert out.shape == x.shape

    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert moe.router.weight.grad is not None
