"""
Correctness stress test under realistic expert imbalance.

Random torch.randn logits (used in all prior tests) are, by construction,
roughly balanced across experts. Real MoE training is not: some experts
become consistently favored ("hot" experts), producing systematically
large-magnitude logits for a few experts and very negative ones for
others — exactly the region where the tanh quantization saturates and
loses resolution. This test specifically stresses that region rather than
the well-behaved random case already covered elsewhere.
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


def make_skewed_logits(N, n_experts, device, skew_strength=8.0, seed=0):
    """Simulate imbalanced routing: a random per-expert bias vector shifts
    some experts to be consistently favored (or disfavored), pushing their
    logits toward the extremes rather than staying near 0."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    expert_bias = (torch.rand(n_experts, generator=g) - 0.5) * 2 * skew_strength  # e.g. [-8, 8]
    base = torch.randn(N, n_experts, generator=g)
    logits = (base + expert_bias.unsqueeze(0)).to(device)
    return logits


@pytest.mark.parametrize("n_experts,k,skew", [
    (64, 8, 8.0),
    (128, 16, 8.0),
    (128, 32, 12.0),   # stronger skew, more saturation pressure
])
def test_fused_route_v2_correct_under_imbalance(n_experts, k, skew):
    from fusedkernels.moe_routing_v2 import fused_moe_route_v2

    N = 2048
    logits_ref = make_skewed_logits(N, n_experts, "cuda", skew_strength=skew).requires_grad_(True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)

    w_ref, idx_ref = naive_route(logits_ref, k)
    w_fused, idx_fused = fused_moe_route_v2(logits_fused, k)

    ref_sets = [set(row.tolist()) for row in idx_ref]
    fused_sets = [set(row.tolist()) for row in idx_fused]
    mismatches = sum(1 for a, b in zip(ref_sets, fused_sets) if a != b)
    assert mismatches == 0, \
        f"{mismatches}/{N} rows selected different experts under skewed logits (skew={skew})"

    def sorted_weights(idx, w):
        order = idx.argsort(dim=-1)
        return torch.gather(w, -1, order)

    w_ref_sorted = sorted_weights(idx_ref, w_ref)
    w_fused_sorted = sorted_weights(idx_fused, w_fused)
    assert torch.allclose(w_ref_sorted, w_fused_sorted, atol=1e-3, rtol=1e-3), \
        "combine weights diverge under skewed/imbalanced routing"


def test_fused_route_v2_extreme_saturation():
    """Push logits to near float32 tanh-saturation extremes directly, to
    probe the boundary of the quantization scheme rather than simulate
    realistic imbalance."""
    from fusedkernels.moe_routing_v2 import fused_moe_route_v2

    torch.manual_seed(0)
    N, n_experts, k = 1024, 64, 8
    logits = torch.randn(N, n_experts, device="cuda") * 2.0
    # force a handful of experts to extreme values in every row
    logits[:, 0] = 50.0    # far past tanh saturation (temp=10 -> tanh(5)=0.9999...)
    logits[:, 1] = -50.0

    w_ref, idx_ref = naive_route(logits.clone(), k)
    w_fused, idx_fused = fused_moe_route_v2(logits.clone(), k)

    ref_sets = [set(row.tolist()) for row in idx_ref]
    fused_sets = [set(row.tolist()) for row in idx_fused]
    assert ref_sets == fused_sets, "extreme-magnitude logits broke top-k selection"

    def sorted_weights(idx, w):
        order = idx.argsort(dim=-1)
        return torch.gather(w, -1, order)

    assert torch.allclose(
        sorted_weights(idx_ref, w_ref), sorted_weights(idx_fused, w_fused),
        atol=1e-3, rtol=1e-3,
    ), "combine weights diverge at extreme logit magnitudes"
