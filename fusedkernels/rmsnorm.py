"""
Fused RMSNorm forward + backward for LLM training.

PyTorch / naive path chains several elementwise ops (pow, mean, rsqrt, mul,
mul) each launching its own CUDA kernel and reading/writing the full
activation tensor from global memory each time. RMSNorm is heavily
memory-bandwidth bound (compute is trivial, tensor is huge), so fusing
into a single kernel that reads the row once and writes it once gives a
close-to-linear speedup with the number of ops fused (typically 3-4x).

This is the same technique used in Liger Kernel / Unsloth / FlashAttention's
LayerNorm kernels.

Usage:
    from fusedkernels.rmsnorm import FusedRMSNorm
    norm = FusedRMSNorm(hidden_size).cuda()
    y = norm(x)   # drop-in replacement for nn.RMSNorm / LlamaRMSNorm
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    x_row_stride, y_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_partial_ptr,
    dy_row_stride, x_row_stride, dx_row_stride, dw_partial_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    dy_row = dy_ptr + row * dy_row_stride
    x_row = x_ptr + row * x_row_stride
    dx_row = dx_ptr + row * dx_row_stride
    dw_row = dw_partial_ptr + row * dw_partial_row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    # dW accumulation (per-row partial, summed on host)
    dw = dy * x * rstd
    tl.store(dw_row + cols, dw, mask=mask)

    # dX = rstd * w * dy - x * rstd^3 / n * sum(dy * w * x)
    wdy = w * dy
    sum_term = tl.sum(wdy * x, axis=0) / n_cols
    dx = rstd * wdy - x * (rstd * rstd * rstd) * sum_term
    tl.store(dx_row + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class _FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1]).contiguous()
        n_rows, n_cols = x.shape

        y = torch.empty_like(x)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        _rmsnorm_fwd_kernel[(n_rows,)](
            x, weight, y, rstd,
            x.stride(0), y.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        dy = dy.reshape(-1, orig_shape[-1]).contiguous()
        n_rows, n_cols = x.shape

        dx = torch.empty_like(x)
        dw_partial = torch.empty((n_rows, n_cols), dtype=torch.float32, device=x.device)

        _rmsnorm_bwd_kernel[(n_rows,)](
            dy, x, weight, rstd, dx, dw_partial,
            dy.stride(0), x.stride(0), dx.stride(0), dw_partial.stride(0),
            n_cols,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        dw = dw_partial.sum(0).to(weight.dtype)
        return dx.reshape(orig_shape), dw, None


class FusedRMSNorm(nn.Module):
    """Drop-in replacement for LlamaRMSNorm / nn.RMSNorm, backed by a fused
    Triton kernel. Same interface: FusedRMSNorm(hidden_size, eps=1e-6)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FusedRMSNormFn.apply(x, self.weight, self.eps)
