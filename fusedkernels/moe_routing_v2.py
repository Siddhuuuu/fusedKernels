"""
v2: fused MoE router using Triton's tl.sort (a bitonic sorting network
under the hood) for top-k selection, plus a vectorized 2D-tile extraction
step. This is the main, recommended router in this library.

Design history (see README for the full narrative):
- v1 (moe_routing.py) selects top-K via K sequential "find max, mask,
  repeat" passes — O(K) serial critical path, degrades badly at high K
  (measured down to 0.18x vs native PyTorch at K=32).
- v2 replaces the sequential scan with ONE tl.sort call (depth
  O(log^2 E), independent of K), then extracts the top-K in one
  vectorized 2D-tile operation instead of a second K-dependent loop.
- Precision fix: ranks by a bounded, monotonic tanh transform of the
  LOGIT (not the softmax probability) before quantizing to an integer
  sort key. Ranking by probability directly breaks down at large expert
  counts because probabilities compress into a tiny range (~1/E on
  average), so quantization error can flip the ranking of near-tied
  experts. Logits aren't compressed this way, so precision holds.
"""

import torch
import triton
import triton.language as tl

_SCALE = 1 << 22
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
    probs = p / s

    # rank by a bounded, monotonic transform of the LOGIT — see module
    # docstring for why this avoids the probability-compression problem
    safe_logits = tl.where(mask, logits, -1e30)
    t = (2.0 / (1.0 + tl.exp(-2.0 * safe_logits / TANH_TEMP))) - 1.0
    quant = ((t + 1.0) * 0.5 * SCALE).to(tl.int32)
    key = tl.where(mask, quant * BLOCK_SIZE + cols, -1)

    # ONE sort call — internally a bitonic network, depth O(log^2 E),
    # independent of K
    sorted_key = tl.sort(key, descending=True)

    # vectorized extraction of all K top elements at once — a
    # (BLOCK_K x BLOCK_SIZE) one-hot tile, reduced in one shot, instead
    # of a K-iteration Python loop
    k_range = tl.arange(0, BLOCK_K)
    kmask = k_range < K
    pos_grid = k_range[:, None] == cols[None, :]
    picked = tl.sum(tl.where(pos_grid, sorted_key[None, :], 0), axis=1)
    idx = picked % BLOCK_SIZE

    # gather the EXACT original probability (not a quantized approximation)
    gather_grid = idx[:, None] == cols[None, :]
    val = tl.sum(tl.where(gather_grid, probs[None, :], 0.0), axis=1)

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
    # subset-softmax Jacobian: the discrete routing DECISION (which
    # experts were picked) is treated as non-differentiable (standard
    # convention, matches Switch Transformer / GShard); only the
    # continuous combine weights carry gradient.
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
    """Fused MoE router: fuses softmax + top-k selection + renormalization
    into one Triton kernel launch, using tl.sort for selection. Drop-in
    replacement for:
        probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_idx = torch.topk(probs, k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    """
    return _FusedMoERouteV2.apply(router_logits, k)
