"""
v1 (BASELINE — kept intentionally, not for production use): fused MoE
router using K sequential "find max, mask it, repeat" passes for top-k
selection.

This is kept in the repo specifically as the naive baseline the v2/v4
benchmark comparisons are measured against — it demonstrates the actual
problem being solved. Its selection step has an O(K) serial critical
path (K sequential reduction passes), which degrades badly as K grows:
measured at 0.18x-0.40x vs native PyTorch (i.e. 2.5-5x SLOWER) at K=32
across various expert counts. See moe_routing_v2.py for the fix.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_route_fwd_kernel(
    logits_ptr, topk_val_ptr, topk_idx_ptr,
    logits_row_stride,
    n_experts,
    K: tl.constexpr,
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

    selected = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    sum_topk = 0.0
    out_val_row = topk_val_ptr + row * K
    out_idx_row = topk_idx_ptr + row * K

    # O(K) sequential passes — each one a full O(log E) reduction — this
    # is the bottleneck v2 fixes.
    for k in range(K):
        cur = tl.where(selected == 0, probs, -1.0)
        idx = tl.argmax(cur, axis=0)
        val = tl.max(cur, axis=0)
        tl.store(out_val_row + k, val)
        tl.store(out_idx_row + k, idx.to(tl.int32))
        selected = tl.where(cols == idx, 1, selected)
        sum_topk += val

    for k in range(K):
        val = tl.load(out_val_row + k)
        tl.store(out_val_row + k, val / sum_topk)


@triton.jit
def _moe_route_bwd_kernel(
    dweight_ptr, topk_val_ptr, topk_idx_ptr, dlogits_ptr,
    n_experts,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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


class _FusedMoERoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, k: int):
        assert logits.is_cuda and logits.ndim == 2
        n_rows, n_experts = logits.shape
        assert k <= n_experts

        logits = logits.contiguous()
        topk_weights = torch.empty((n_rows, k), dtype=torch.float32, device=logits.device)
        topk_idx = torch.empty((n_rows, k), dtype=torch.int32, device=logits.device)

        BLOCK_SIZE = triton.next_power_of_2(n_experts)
        _moe_route_fwd_kernel[(n_rows,)](
            logits, topk_weights, topk_idx,
            logits.stride(0),
            n_experts,
            K=k,
            BLOCK_SIZE=BLOCK_SIZE,
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

        _moe_route_bwd_kernel[(n_rows,)](
            dweights.contiguous(), topk_weights, topk_idx, dlogits,
            n_experts,
            K=k,
            BLOCK_K=BLOCK_K,
        )
        return dlogits, None


def fused_moe_route(router_logits: torch.Tensor, k: int):
    """v1 baseline router — see module docstring. Prefer fused_moe_route_v2."""
    return _FusedMoERoute.apply(router_logits, k)
