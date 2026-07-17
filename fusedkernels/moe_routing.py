"""
Fused MoE (Mixture-of-Experts) router.

Standard MoE routing (used in Mixtral, DeepSeek-MoE, Qwen-MoE, Switch
Transformer, etc.) looks like this in almost every open implementation:

    probs = F.softmax(router_logits, dim=-1)       # kernel launch 1
    topk_weights, topk_idx = torch.topk(probs, k)  # kernel launch 2
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)  # launch 3

Each step reads/writes a [N, E] or [N, K] tensor from global memory. E
(number of experts) is usually small (8-128), so this op is *launch-overhead
and memory-traffic bound*, not compute bound — exactly the kind of op that
benefits from fusion, since the actual math is trivial but it's called on
every single token, every single layer, every forward pass.

This kernel fuses softmax + top-k selection + renormalization into ONE
kernel launch (forward), with a matching fused backward.

Math note on the backward: renormalizing a full softmax restricted to a
top-k subset is mathematically identical to computing softmax directly on
just that subset's logits (the normalization constant cancels). So the
backward is a standard K x K softmax Jacobian restricted to the selected
experts; gradient w.r.t. non-selected experts' logits is zero. This
matches the standard convention used in Switch Transformer / GShard: the
*discrete* routing decision (which experts get picked) is treated as
non-differentiable, only the continuous combine weights carry gradient.

Usage:
    from fusedkernels.moe_routing import fused_moe_route
    topk_weights, topk_idx = fused_moe_route(router_logits, k=2)
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

    # full softmax over experts (numerically stable)
    m = tl.max(logits, axis=0)
    p = tl.exp(logits - m)
    p = tl.where(mask, p, 0.0)
    s = tl.sum(p, axis=0)
    probs = p / s

    # iterative top-k: repeatedly take the argmax of what's left, mask it out
    selected = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    sum_topk = 0.0
    out_val_row = topk_val_ptr + row * K
    out_idx_row = topk_idx_ptr + row * K

    for k in range(K):
        cur = tl.where(selected == 0, probs, -1.0)  # probs are >=0, so -1 is a safe "excluded" sentinel
        idx = tl.argmax(cur, axis=0)
        val = tl.max(cur, axis=0)
        tl.store(out_val_row + k, val)
        tl.store(out_idx_row + k, idx.to(tl.int32))
        selected = tl.where(cols == idx, 1, selected)
        sum_topk += val

    # renormalize the K selected weights to sum to 1
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

    # standard softmax-Jacobian restricted to the K selected experts
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
    """Fused equivalent of:
        probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_idx = torch.topk(probs, k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    Args:
        router_logits: [N, E] raw router logits (pre-softmax)
        k: number of experts to route each token to

    Returns:
        topk_weights: [N, K] float32, normalized combine weights (sum to 1 per row)
        topk_idx: [N, K] long, indices of selected experts per row
    """
    return _FusedMoERoute.apply(router_logits, k)
