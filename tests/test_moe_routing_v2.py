"""
Correctness tests for the experimental bitonic-sort router (v2).
Same checks as test_moe_correctness.py, run against v2 instead of v1.

NOTE: this kernel is new and untested on real hardware as of writing.
If any of these fail, the error output is exactly what's needed to debug
the tl.sort usage / key-packing logic — paste it back for a fix.
"""

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


@pytest.mark.parametrize("n_experts,k", [(8, 2), (16, 4), (64, 8), (128, 16)])
def test_fused_moe_route_v2_matches_naive(n_experts, k):
    from fusedkernels.moe_routing_v2 import fused_moe_route_v2

    torch.manual_seed(0)
    N = 1024
    logits_ref = torch.randn(N, n_experts, device="cuda", requires_grad=True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)

    w_ref, idx_ref = naive_route(logits_ref, k)
    w_fused, idx_fused = fused_moe_route_v2(logits_fused, k)

    ref_sets = [set(row.tolist()) for row in idx_ref]
    fused_sets = [set(row.tolist()) for row in idx_fused]
    assert ref_sets == fused_sets, "v2: selected expert sets differ from naive routing"

    def sorted_weights(idx, w):
        order = idx.argsort(dim=-1)
        return torch.gather(w, -1, order)

    w_ref_sorted = sorted_weights(idx_ref, w_ref)
    w_fused_sorted = sorted_weights(idx_fused, w_fused)
    assert torch.allclose(w_ref_sorted, w_fused_sorted, atol=1e-3, rtol=1e-3), \
        "v2: combine weights differ from naive routing"

    grad_sorted = torch.randn_like(w_ref_sorted)

    def unsort_like(idx, sorted_vals):
        order = idx.argsort(dim=-1)
        inv_order = order.argsort(dim=-1)
        return torch.gather(sorted_vals, -1, inv_order)

    grad_ref = unsort_like(idx_ref, grad_sorted)
    grad_fused = unsort_like(idx_fused, grad_sorted)

    w_ref.backward(grad_ref)
    w_fused.backward(grad_fused)

    assert torch.allclose(logits_ref.grad, logits_fused.grad, atol=1e-3, rtol=1e-3), \
        "v2: router logits gradient mismatch vs naive routing"


def test_fused_moe_route_v2_weights_sum_to_one():
    from fusedkernels.moe_routing_v2 import fused_moe_route_v2

    torch.manual_seed(1)
    logits = torch.randn(512, 32, device="cuda")
    w, _ = fused_moe_route_v2(logits, k=4)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        "v2: top-k combine weights should sum to 1 per token"


@pytest.mark.parametrize("n_experts,k", [(64, 8), (128, 16)])
def test_v2_matches_v1(n_experts, k):
    """Sanity cross-check: v1 (sequential) and v2 (sorted) should select the
    same experts and weights on the same input, despite different algorithms."""
    from fusedkernels.moe_routing import fused_moe_route
    from fusedkernels.moe_routing_v2 import fused_moe_route_v2

    torch.manual_seed(2)
    logits = torch.randn(256, n_experts, device="cuda")

    w1, idx1 = fused_moe_route(logits.clone(), k)
    w2, idx2 = fused_moe_route_v2(logits.clone(), k)

    sets1 = [set(row.tolist()) for row in idx1]
    sets2 = [set(row.tolist()) for row in idx2]
    assert sets1 == sets2, "v1 and v2 disagree on selected experts"
