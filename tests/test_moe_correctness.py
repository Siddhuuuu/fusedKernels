"""
Correctness tests for the fused MoE router: compare against the standard
naive implementation (softmax -> topk -> renormalize) that most open MoE
codebases use, checking both forward output and backward gradients.
"""

import pytest
import torch
import torch.nn.functional as F

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA GPU")


def naive_route(logits, k):
    """Reference implementation: the standard 3-op MoE routing path."""
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

    # indices should select the same set of experts per token (order may
    # differ since ties/argmax order aren't guaranteed identical, so we
    # compare as sets per row)
    ref_sets = [set(row.tolist()) for row in idx_ref]
    fused_sets = [set(row.tolist()) for row in idx_fused]
    assert ref_sets == fused_sets, "selected expert sets differ between naive and fused routing"

    # compare weights after sorting each row's (idx, weight) pairs by idx,
    # since the two implementations may order the top-k differently
    def sorted_weights(idx, w):
        order = idx.argsort(dim=-1)
        return torch.gather(w, -1, order)

    w_ref_sorted = sorted_weights(idx_ref, w_ref)
    w_fused_sorted = sorted_weights(idx_fused, w_fused)
    assert torch.allclose(w_ref_sorted, w_fused_sorted, atol=1e-3, rtol=1e-3), \
        "combine weights differ between naive and fused routing"

    # backward: use the same upstream grad in the same (idx-sorted) order
    grad_sorted = torch.randn_like(w_ref_sorted)
    # unsort grad back into each implementation's own idx order before backward
    def unsort_like(idx, sorted_vals, orig_idx):
        order = idx.argsort(dim=-1)
        inv_order = order.argsort(dim=-1)
        return torch.gather(sorted_vals, -1, inv_order)

    grad_ref = unsort_like(idx_ref, grad_sorted, idx_ref)
    grad_fused = unsort_like(idx_fused, grad_sorted, idx_fused)

    w_ref.backward(grad_ref)
    w_fused.backward(grad_fused)

    assert torch.allclose(logits_ref.grad, logits_fused.grad, atol=1e-3, rtol=1e-3), \
        "router logits gradient mismatch between naive and fused routing"


def test_fused_moe_route_weights_sum_to_one():
    from fusedkernels.moe_routing import fused_moe_route

    torch.manual_seed(1)
    logits = torch.randn(512, 32, device="cuda")
    w, _ = fused_moe_route(logits, k=4)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        "top-k combine weights should sum to 1 per token"


def test_fused_moe_mlp_runs_and_shapes_match():
    from fusedkernels.moe_layer import FusedMoEMLP

    torch.manual_seed(2)
    B, T, D = 2, 64, 256
    moe = FusedMoEMLP(dim=D, hidden_dim=512, n_experts=8, k=2).cuda()
    x = torch.randn(B, T, D, device="cuda", requires_grad=True)

    out = moe(x)
    assert out.shape == x.shape

    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    # router weight should have received gradient too
    assert moe.router.weight.grad is not None
