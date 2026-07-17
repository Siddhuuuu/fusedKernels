"""
Fused Cross-Entropy loss for LLM training.

Standard PyTorch path:
    logits = model(x)                # [B*T, V]
    loss = F.cross_entropy(logits, targets)

materializes softmax probabilities over the full vocab for every token,
i.e. an extra [B*T, V] fp32 tensor. For V=128k and B*T=32k tokens that's
~16GB just for one intermediate. This kernel computes the per-row loss
AND the gradient w.r.t. logits in a single pass over each row, using
online (streaming) softmax so the full [B*T, V] probability tensor is
never materialized — only the final loss (scalar-ish) and the gradient
(same size as the logits, which you needed anyway) are stored.

This mirrors the approach used by Liger Kernel / Unsloth for their fused
CE loss, which reduces peak activation memory substantially and speeds
up the loss step by avoiding multiple elementwise kernel launches.

Usage:
    from fusedkernels.cross_entropy import fused_cross_entropy
    loss = fused_cross_entropy(logits, targets, ignore_index=-100)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_fwd_bwd_kernel(
    logits_ptr, targets_ptr, loss_ptr, grad_ptr,
    logits_row_stride, grad_row_stride,
    n_cols,
    ignore_index,
    label_smoothing,
    BLOCK_SIZE: tl.constexpr,
):
    """One program instance handles one row (one token's logits over vocab)."""
    row_idx = tl.program_id(0)
    logits_row_ptr = logits_ptr + row_idx * logits_row_stride
    grad_row_ptr = grad_ptr + row_idx * grad_row_stride

    target = tl.load(targets_ptr + row_idx)

    # --- Pass 1: online max + sum(exp) for numerically stable softmax ---
    m = -float("inf")
    s = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(logits_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m, block_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m), axis=0)
        m = new_m

    log_sum_exp = m + tl.log(s)

    is_ignored = target == ignore_index

    # --- loss = log_sum_exp - logit[target] (+ label smoothing term) ---
    target_logit = tl.load(logits_row_ptr + target, mask=(~is_ignored), other=0.0).to(tl.float32)
    loss = log_sum_exp - target_logit

    if label_smoothing > 0.0:
        # smoothing pulls loss toward the uniform-over-vocab distribution
        smooth_loss = log_sum_exp - (1.0 / n_cols) * 0.0  # placeholder term computed below
        sum_logits = 0.0
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            vals = tl.load(logits_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            sum_logits += tl.sum(vals, axis=0)
        mean_logit = sum_logits / n_cols
        smooth_loss = log_sum_exp - mean_logit
        loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth_loss

    loss = tl.where(is_ignored, 0.0, loss)
    tl.store(loss_ptr + row_idx, loss)

    # --- Pass 2: gradient = softmax(logits) - one_hot(target) (scaled) ---
    smooth_term = label_smoothing / n_cols
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(logits_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.exp(vals - m) / s
        target_mask = cols == target
        grad = p - (1.0 - label_smoothing) * tl.where(target_mask, 1.0, 0.0) - smooth_term
        grad = tl.where(is_ignored, 0.0, grad)
        tl.store(grad_row_ptr + cols, grad.to(grad_ptr.dtype.element_ty), mask=mask)


class _FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor,
                ignore_index: int = -100, label_smoothing: float = 0.0):
        assert logits.is_cuda and logits.ndim == 2
        n_rows, n_cols = logits.shape
        logits = logits.contiguous()
        targets = targets.contiguous()

        loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad = torch.empty_like(logits)

        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 16384)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)

        _cross_entropy_fwd_bwd_kernel[(n_rows,)](
            logits, targets, loss, grad,
            logits.stride(0), grad.stride(0),
            n_cols,
            ignore_index,
            label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        n_valid = (targets != ignore_index).sum().clamp(min=1)
        ctx.save_for_backward(grad)
        ctx.n_valid = n_valid
        return loss.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        # grad_output is a scalar (dLoss/dLoss_mean); scale precomputed per-row grad
        scale = grad_output / ctx.n_valid
        return grad * scale, None, None, None


def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         ignore_index: int = -100, label_smoothing: float = 0.0) -> torch.Tensor:
    """Drop-in fused replacement for F.cross_entropy(logits, targets).

    logits: [N, V] float tensor (pre-softmax)
    targets: [N] long tensor of class indices, or ignore_index to skip a row
    Returns: scalar mean loss over non-ignored rows.
    """
    return _FusedCrossEntropy.apply(logits, targets, ignore_index, label_smoothing)
