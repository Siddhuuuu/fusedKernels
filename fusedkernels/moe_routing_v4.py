"""
v4 (EXPERIMENTAL): a truncated/grouped bitonic selection network for MoE
routing — inspired by the technique FAISS uses for GPU top-k selection
(Johnson et al., 2017), adapted here to Triton for MoE router combine
weights specifically.

Motivation: v2 fuses selection into ONE tl.sort call, but a full sort of
all E experts does strictly more work than a top-K SELECTION needs
(O(E log^2 E) vs the O(E) theoretical optimum for selection). v4 instead:
  1. Locally sorts small groups of size KP = next_pow2(K)
     (_bitonic_primitive.bitonic_sort_bounded)
  2. Repeatedly merges pairs of groups' top-KP lists
     (_bitonic_merge_primitive.bitonic_merge_topk), discarding everything
     except the top-KP after each merge — so later stages only ever
     process KP-scale data, never the full E-sized array.

This targets O(E log^2 K) work instead of O(E log^2 E) — meaningful when
K << E, which is the realistic MoE regime (Mixtral K=2/E=8,
DeepSeek-MoE K=6-8/E=64+).

STATUS: correct and deterministic for the tested range (K up to 32,
E up to 128) after fixing a real bug in an earlier hand-rolled merge
implementation — see DEBUGGING.md for the investigation, and
_bitonic_merge_primitive.py's docstring for the specific fix. Composed
via Python orchestration (calling the verified kernels in sequence) — an
honest trade-off of some kernel-launch overhead for much higher
confidence than a single monolithic kernel with hand-rolled multi-level
bookkeeping.
"""

import torch
import triton
import triton.language as tl
import warnings

from ._bitonic_primitive import bitonic_sort_bounded
from ._bitonic_merge_primitive import bitonic_merge_topk
from .moe_routing_v2 import _moe_route_bwd_kernel_v2

_SCALE = 1 << 22
_TANH_TEMP = 10.0


@triton.jit
def _compute_keys_and_probs_kernel(
    logits_ptr, probs_ptr, keys_ptr,
    logits_row_stride, probs_row_stride, keys_row_stride,
    n_experts,
    SCALE: tl.constexpr,
    TANH_TEMP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_experts

    logits_row = logits_ptr + row * logits_row_stride
    logits = tl.load(logits_row + cols, mask=mask, other=-float("inf")).to(tl.float32)

    m = tl.max(logits, axis=0)
    p = tl.exp(logits - m)
    p = tl.where(mask, p, 0.0)
    s = tl.sum(p, axis=0)
    probs = p / s

    safe_logits = tl.where(mask, logits, -1e30)
    t = (2.0 / (1.0 + tl.exp(-2.0 * safe_logits / TANH_TEMP))) - 1.0
    quant = ((t + 1.0) * 0.5 * SCALE).to(tl.int32)
    key = quant * BLOCK_SIZE + cols

    probs_row = probs_ptr + row * probs_row_stride
    keys_row = keys_ptr + row * keys_row_stride
    tl.store(probs_row + cols, probs, mask=mask)
    tl.store(keys_row + cols, key, mask=mask)


def _compute_keys_and_probs(logits: torch.Tensor):
    n_rows, n_experts = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(n_experts)
    logits = logits.contiguous()

    probs = torch.empty((n_rows, n_experts), dtype=torch.float32, device=logits.device)
    keys = torch.empty((n_rows, n_experts), dtype=torch.int32, device=logits.device)

    _compute_keys_and_probs_kernel[(n_rows,)](
        logits, probs, keys,
        logits.stride(0), probs.stride(0), keys.stride(0),
        n_experts,
        SCALE=_SCALE,
        TANH_TEMP=_TANH_TEMP,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return probs, keys


def _select_topk_bitonic(logits: torch.Tensor, k: int):
    n_rows, n_experts = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(n_experts)
    KP = triton.next_power_of_2(k)
    assert BLOCK_SIZE % KP == 0

    probs, keys = _compute_keys_and_probs(logits)

    sorted_keys = bitonic_sort_bounded(keys, KP)
    candidates = sorted_keys.view(n_rows, BLOCK_SIZE // KP, KP)

    while candidates.shape[1] > 1:
        n_groups = candidates.shape[1]
        A = candidates[:, 0::2, :].reshape(-1, KP)
        B = candidates[:, 1::2, :].reshape(-1, KP)
        merged = bitonic_merge_topk(A, B)
        candidates = merged.view(n_rows, n_groups // 2, KP)

    final_keys = candidates.view(n_rows, KP)
    topk_keys = final_keys[:, KP - k:]

    idx = (topk_keys % BLOCK_SIZE).long()
    weight_raw = torch.gather(probs, 1, idx)
    sum_topk = weight_raw.sum(dim=-1, keepdim=True)
    weight = weight_raw / sum_topk
    return weight, idx


class _FusedMoERouteV4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, k: int):
        assert logits.is_cuda and logits.ndim == 2
        n_rows, n_experts = logits.shape
        assert k <= n_experts

        weight, idx = _select_topk_bitonic(logits, k)

        ctx.save_for_backward(weight, idx.to(torch.int32))
        ctx.n_experts = n_experts
        ctx.k = k
        ctx.logits_shape = logits.shape
        return weight, idx

    @staticmethod
    def backward(ctx, dweights, dindices_unused):
        topk_weights, topk_idx = ctx.saved_tensors
        n_rows, n_experts = ctx.logits_shape
        k = ctx.k

        dlogits = torch.zeros((n_rows, n_experts), dtype=torch.float32, device=topk_weights.device)
        BLOCK_K = triton.next_power_of_2(k)

        _moe_route_bwd_kernel_v2[(n_rows,)](
            dweights.contiguous(), topk_weights, topk_idx, dlogits,
            n_experts,
            K=k,
            BLOCK_K=BLOCK_K,
        )
        return dlogits, None


def fused_moe_route_v4(router_logits: torch.Tensor, k: int):
    """v4 (experimental): truncated bitonic selection network router.
    Same interface as fused_moe_route_v2. See module docstring for design
    and status."""
    n_experts = router_logits.shape[-1]
    if k > n_experts // 2:
        warnings.warn(
            f"fused_moe_route_v4: k={k} is a large fraction of n_experts={n_experts}. "
            "Tested and correct up to k=32 at e=128 after the merge-kernel fix "
            "(see DEBUGGING.md); real MoE configs (k << n_experts) are well within "
            "the validated range.",
            stacklevel=2,
        )
    return _FusedMoERouteV4.apply(router_logits, k)
