"""
EXPERIMENTAL: bitonic-sort-based fused MoE router.

moe_routing.py's kernel selects top-K via K sequential "find max, mask it,
repeat" passes. That has a serial critical path of depth O(K) — fine for
K=2, but by K=8 with many experts it starts losing to PyTorch's topk,
which uses a more parallel selection algorithm.

This version replaces the sequential scan with ONE sort, using Triton's
`tl.sort`, which is itself implemented as a bitonic sorting network under
the hood — the same recursive butterfly/divide-and-conquer structure as
an FFT (compare-and-swap stages instead of butterfly adds). A bitonic sort
of E elements has depth O(log^2 E), independent of K, so this should scale
much better as K grows.

The catch: tl.sort only sorts a vector of values, it doesn't carry a
companion index array. So we pack (quantized_key, original_index) into a
single sortable int32 key before sorting, then unpack after.

v1 -> v2 fix #1: the first version quantized the *softmax probability*.
That breaks down at large expert counts: probabilities get compressed into
a tiny range (~1/E on average), so many experts near the top-K boundary
can differ by less than the quantization step, causing occasional wrong
selections at the cutoff (confirmed by testing: correct at E=64, wrong at
E=128,k=16). Fix: quantize the *logit* instead (via a bounded, strictly
monotonic tanh transform), since softmax is monotonic in the logit and
logits aren't compressed into [0, 1/E] the way probabilities are.

v2 fix #2: after fixing correctness, benchmarking revealed forward time
still scaled ~linearly with K, almost as steeply as the original
sequential kernel. Root cause: extracting the top-K elements out of the
sorted array was *itself* still a `for kk in range(K)` Python loop — the
sort became K-independent, but the readout after it wasn't. Fixed by
extracting all K elements in one vectorized 2D operation: build a
(BLOCK_K x BLOCK_SIZE) comparison tile and reduce once, instead of K
separate sequential reductions.

STATUS: this is a new, less battle-tested kernel than the rest of the
library. Run tests/test_moe_routing_v2.py and report any errors.

Usage:
    from fusedkernels.moe_routing_v2 import fused_moe_route_v2
    topk_weights, topk_idx = fused_moe_route_v2(router_logits, k=8)
"""

import torch
import triton
import triton.language as tl

# quantization scale for packing the (tanh-squashed) logit into the sort
# key. tanh output is in (-1, 1); SCALE * BLOCK_SIZE must stay well under
# int32 range (~2.1e9).
_SCALE = 1 << 22
# temperature for the tanh squashing — larger T means less saturation for
# large-magnitude logits, at the cost of slightly less resolution near 0.
# 10.0 comfortably covers typical router logit magnitudes without
# saturating in the region that matters for top-k ranking.
_TANH_TEMP = 10.0


@triton.jit
def _moe_route_fwd_kernel_v2(
    logits_ptr, topk_val_ptr, topk_idx_ptr,
    logits_row_stride,
    n_experts,
    SCALE: tl.constexpr,
    TANH_TEMP: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_experts

    logits_row = logits_ptr + row * logits_row_stride
    logits = tl.load(logits_row + cols, mask=mask, other=-float("inf")).to(tl.float32)

    # softmax over experts (for the final normalized combine weight)
    m = tl.max(logits, axis=0)
    p = tl.exp(logits - m)
    p = tl.where(mask, p, 0.0)
    s = tl.sum(p, axis=0)
    probs = p / s  # padding slots -> 0 (since their logits were -inf)

    # rank by a bounded, strictly-monotonic transform of the LOGIT (not the
    # probability) — avoids the probability-compression precision problem
    # at large expert counts, since softmax(x) is monotonic in x.
    safe_logits = tl.where(mask, logits, -1e30)  # keep padding maximally negative pre-tanh
    t = (2.0 / (1.0 + tl.exp(-2.0 * safe_logits / TANH_TEMP))) - 1.0  # tanh(x/T) via sigmoid identity
    quant = ((t + 1.0) * 0.5 * SCALE).to(tl.int32)  # in [0, SCALE]
    key = tl.where(mask, quant * BLOCK_SIZE + cols, -1)  # padding sorts last

    # ONE sort call — internally a bitonic network, depth O(log^2 BLOCK_SIZE),
    # independent of K. This replaces the K sequential comparison passes.
    sorted_key = tl.sort(key, descending=True)

    # --- vectorized extraction of all K top elements at once ---
    # build a (BLOCK_K x BLOCK_SIZE) tile: row kk picks out position kk of
    # sorted_key via a one-hot mask, all K rows computed in parallel instead
    # of a sequential kk=0..K-1 loop.
    k_range = tl.arange(0, BLOCK_K)
    kmask = k_range < K
    pos_grid = k_range[:, None] == cols[None, :]           # (BLOCK_K, BLOCK_SIZE)
    picked = tl.sum(tl.where(pos_grid, sorted_key[None, :], 0), axis=1)  # (BLOCK_K,)
    idx = picked % BLOCK_SIZE                               # (BLOCK_K,)

    # vectorized gather of the exact original probability for each picked
    # index (not a decoded/quantized approximation) — same one-hot-grid
    # trick, this time gathering from `probs` instead of `sorted_key`.
    gather_grid = idx[:, None] == cols[None, :]              # (BLOCK_K, BLOCK_SIZE)
    val = tl.sum(tl.where(gather_grid, probs[None, :], 0.0), axis=1)  # (BLOCK_K,)

    sum_topk = tl.sum(tl.where(kmask, val, 0.0), axis=0)
    val_normalized = val / sum_topk

    out_val_row = topk_val_ptr + row * K
    out_idx_row = topk_idx_ptr + row * K
    tl.store(out_val_row + k_range, val_normalized, mask=kmask)
    tl.store(out_idx_row + k_range, idx.to(tl.int32), mask=kmask)


@triton.jit
def _moe_route_bwd_kernel_v2(
    dweight_ptr, topk_val_ptr, topk_idx_ptr, dlogits_ptr,
    n_experts,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # identical backward math to v1 — subset-softmax Jacobian, unaffected by
    # how the forward selected the top-K, only by which K were selected.
    row = tl.program_id(0)
    k_range = tl.arange(0, BLOCK_K)
    kmask = k_range < K
    base = row * K

    w = tl.load(topk_val_ptr + base + k_range, mask=kmask, other=0.0)
    dw = tl.load(dweight_ptr + base + k_range, mask=kmask, other=0.0)
    idx = tl.load(topk_idx_ptr + base + k_range, mask=kmask, other=0)

    dot = tl.sum(w * dw, axis=0)
    dlogits_topk = w * (dw - dot)

    dlogits_row = dlogits_ptr + row * n_experts
    tl.store(dlogits_row + idx, dlogits_topk, mask=kmask)


class _FusedMoERouteV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, k: int):
        assert logits.is_cuda and logits.ndim == 2
        n_rows, n_experts = logits.shape
        assert k <= n_experts

        BLOCK_SIZE = triton.next_power_of_2(n_experts)
        BLOCK_K = triton.next_power_of_2(k)
        assert _SCALE * BLOCK_SIZE < (1 << 31), \
            "SCALE * BLOCK_SIZE would overflow int32 — reduce _SCALE for this expert count"

        logits = logits.contiguous()
        topk_weights = torch.empty((n_rows, k), dtype=torch.float32, device=logits.device)
        topk_idx = torch.empty((n_rows, k), dtype=torch.int32, device=logits.device)

        _moe_route_fwd_kernel_v2[(n_rows,)](
            logits, topk_weights, topk_idx,
            logits.stride(0),
            n_experts,
            SCALE=_SCALE,
            TANH_TEMP=_TANH_TEMP,
            K=k,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(topk_weights, topk_idx)
        ctx.n_experts = n_experts
        ctx.k = k
        ctx.logits_shape = logits.shape
        return topk_weights, topk_idx.long()

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


def fused_moe_route_v2(router_logits: torch.Tensor, k: int):
    """Experimental bitonic-sort-based version of fused_moe_route.
    Same interface and semantics as fused_moe_route (see moe_routing.py),
    but selects top-K via a single sort instead of K sequential passes —
    intended to scale better as K grows.
    """
    return _FusedMoERouteV2.apply(router_logits, k)
