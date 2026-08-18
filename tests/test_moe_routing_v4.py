"""
Correctness tests for v4 (experimental truncated bitonic router), now
using the FIXED tl.sort-based merge. Includes a regression test for the
exact non-determinism bug that was found and fixed.
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


@pytest.mark.parametrize("n_experts,k", [
    (8, 2), (16, 4), (16, 8), (32, 4), (32, 8), (32, 16),
    (64, 8), (64, 16), (64, 32), (128, 8), (128, 16), (128, 32),
])
def test_fused_moe_route_v4_matches_naive(n_experts, k):
    from fusedkernels.moe_routing_v4 import fused_moe_route_v4

    torch.manual_seed(0)
    N = 1024
    logits_ref = torch.randn(N, n_experts, device="cuda", requires_grad=True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)

    w_ref, idx_ref = naive_route(logits_ref, k)
    w_fused, idx_fused = fused_moe_route_v4(logits_fused, k)

    ref_sets = [set(row.tolist()) for row in idx_ref]
    fused_sets = [set(row.tolist()) for row in idx_fused]
    mismatches = sum(1 for a, b in zip(ref_sets, fused_sets) if a != b)
    assert mismatches == 0, \
        f"{mismatches}/{N} rows selected different experts (v4, n_experts={n_experts}, k={k})"

    def sorted_weights(idx, w):
        order = idx.argsort(dim=-1)
        return torch.gather(w, -1, order)

    w_ref_sorted = sorted_weights(idx_ref, w_ref)
    w_fused_sorted = sorted_weights(idx_fused, w_fused)
    assert torch.allclose(w_ref_sorted, w_fused_sorted, atol=1e-3, rtol=1e-3)

    grad_sorted = torch.randn_like(w_ref_sorted)

    def unsort_like(idx, sorted_vals):
        order = idx.argsort(dim=-1)
        inv_order = order.argsort(dim=-1)
        return torch.gather(sorted_vals, -1, inv_order)

    grad_ref = unsort_like(idx_ref, grad_sorted)
    grad_fused = unsort_like(idx_fused, grad_sorted)

    w_ref.backward(grad_ref)
    w_fused.backward(grad_fused)

    assert torch.allclose(logits_ref.grad, logits_fused.grad, atol=1e-3, rtol=1e-3)


def test_fused_moe_route_v4_weights_sum_to_one():
    from fusedkernels.moe_routing_v4 import fused_moe_route_v4

    torch.manual_seed(1)
    logits = torch.randn(512, 64, device="cuda")
    w, _ = fused_moe_route_v4(logits, k=8)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_fused_moe_route_v4_non_power_of_2_k():
    from fusedkernels.moe_routing_v4 import fused_moe_route_v4

    torch.manual_seed(2)
    for n_experts, k in [(32, 3), (64, 5), (128, 7), (64, 11)]:
        logits = torch.randn(256, n_experts, device="cuda", requires_grad=True)
        logits_ref = logits.detach().clone().requires_grad_(True)

        w_ref, idx_ref = naive_route(logits_ref, k)
        w_fused, idx_fused = fused_moe_route_v4(logits, k)

        assert w_fused.shape == (256, k)
        ref_sets = [set(row.tolist()) for row in idx_ref]
        fused_sets = [set(row.tolist()) for row in idx_fused]
        assert ref_sets == fused_sets


def test_fused_moe_route_v4_matches_v2():
    from fusedkernels.moe_routing_v2 import fused_moe_route_v2
    from fusedkernels.moe_routing_v4 import fused_moe_route_v4

    torch.manual_seed(3)
    for n_experts, k in [(64, 8), (128, 16), (128, 32)]:
        logits = torch.randn(256, n_experts, device="cuda")

        w2, idx2 = fused_moe_route_v2(logits.clone(), k)
        w4, idx4 = fused_moe_route_v4(logits.clone(), k)

        sets2 = [set(row.tolist()) for row in idx2]
        sets4 = [set(row.tolist()) for row in idx4]
        assert sets2 == sets4, f"v2 and v4 disagree at n_experts={n_experts}, k={k}"


def test_fused_moe_route_v4_deterministic_at_scale():
    """Regression test for the specific bug found and fixed: v4's earlier
    hand-rolled merge kernel gave different results across repeated calls
    on IDENTICAL large input (215-233/1024 differing rows). This test
    would have caught that bug directly."""
    from fusedkernels.moe_routing_v4 import fused_moe_route_v4

    torch.manual_seed(4)
    N, n_experts, k = 1024, 128, 32
    logits = torch.randn(N, n_experts, device="cuda")

    first_idx = None
    for trial in range(5):
        _, idx = fused_moe_route_v4(logits.clone(), k)
        if first_idx is None:
            first_idx = idx.clone()
        else:
            assert torch.equal(idx, first_idx), \
                f"non-deterministic at trial {trial}: v4 gave different results on identical input"
