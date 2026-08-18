"""
Bounded bitonic sort: independently ascending-sorts each contiguous
STOP_K-sized chunk of a row. Used as phase 1 of the truncated selection
network in moe_routing_v4.py (locally sort small groups before merging
them, instead of fully sorting the whole expert array).

Verified against torch.sort ground truth across power-of-2 and
non-power-of-2 sizes, duplicate values, and group-independence — see
tests/test_bitonic_primitive.py and tests/test_bitonic_bounded_sort.py.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_sort_ascending_kernel(
    x_ptr, scratch_ptr, out_ptr,
    row_stride, scratch_row_stride, out_row_stride,
    n_elements,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    mask = cols < n_elements

    x_row = x_ptr + row * row_stride
    vals = tl.load(x_row + cols, mask=mask, other=2147483647)

    scratch_row = scratch_ptr + row * scratch_row_stride
    tl.store(scratch_row + cols, vals)

    k = 2
    while k <= N:
        j = k // 2
        while j >= 1:
            cur = tl.load(scratch_row + cols)
            partner_idx = cols ^ j
            partner_val = tl.load(scratch_row + partner_idx)

            is_lower = (cols & j) == 0
            ascending_block = (cols & k) == 0
            take_min = is_lower == ascending_block

            new_val = tl.where(take_min, tl.minimum(cur, partner_val), tl.maximum(cur, partner_val))

            tl.store(scratch_row + cols, new_val)
            tl.debug_barrier()

            j = j // 2
        k = k * 2

    result = tl.load(scratch_row + cols, mask=mask)
    out_row = out_ptr + row * out_row_stride
    tl.store(out_row + cols, result, mask=mask)


def bitonic_sort_ascending(x: torch.Tensor) -> torch.Tensor:
    """Sort each row of x ascending using a hand-written bitonic network.
    x: [n_rows, n_elements] int32. Verification utility (see tests)."""
    assert x.is_cuda and x.ndim == 2 and x.dtype == torch.int32
    n_rows, n_elements = x.shape
    N = triton.next_power_of_2(n_elements)

    x = x.contiguous()
    scratch = torch.empty((n_rows, N), dtype=torch.int32, device=x.device)
    out = torch.empty((n_rows, n_elements), dtype=torch.int32, device=x.device)

    _bitonic_sort_ascending_kernel[(n_rows,)](
        x, scratch, out,
        x.stride(0), scratch.stride(0), out.stride(0),
        n_elements,
        N=N,
    )
    return out


@triton.jit
def _bitonic_sort_bounded_kernel(
    x_ptr, scratch_ptr, out_ptr,
    row_stride, scratch_row_stride, out_row_stride,
    n_elements,
    N: tl.constexpr,
    STOP_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    mask = cols < n_elements

    x_row = x_ptr + row * row_stride
    vals = tl.load(x_row + cols, mask=mask, other=-1073741824)

    scratch_row = scratch_ptr + row * scratch_row_stride
    tl.store(scratch_row + cols, vals)

    k = 2
    while k <= STOP_K:
        j = k // 2
        while j >= 1:
            cur = tl.load(scratch_row + cols)
            partner_idx = cols ^ j
            partner_val = tl.load(scratch_row + partner_idx)

            is_lower = (cols & j) == 0
            # Alternating direction by block position IS mathematically
            # required here — this is what makes each level's input a
            # valid bitonic sequence for the next level. Do not remove.
            ascending_block = (cols & k) == 0
            take_min = is_lower == ascending_block

            new_val = tl.where(take_min, tl.minimum(cur, partner_val), tl.maximum(cur, partner_val))

            tl.store(scratch_row + cols, new_val)
            tl.debug_barrier()

            j = j // 2
        k = k * 2

    # Each STOP_K-sized group is now correctly sorted, but groups
    # ALTERNATE ascending/descending (unavoidable side effect of the
    # alternation above). Fix up: reverse any group that ended up
    # descending, so every group is independently ascending.
    local_pos = cols & (STOP_K - 1)
    group_start = cols - local_pos
    reversed_idx = group_start + (STOP_K - 1) - local_pos

    is_descending_group = (cols & STOP_K) != 0
    cur = tl.load(scratch_row + cols)
    reversed_val = tl.load(scratch_row + reversed_idx)
    final_val = tl.where(is_descending_group, reversed_val, cur)

    out_row = out_ptr + row * out_row_stride
    tl.store(out_row + cols, final_val)


def bitonic_sort_bounded(x: torch.Tensor, stop_k: int) -> torch.Tensor:
    """Independently ascending-sorts each contiguous stop_k-sized chunk of
    every row. x: [n_rows, n_elements] int32, stop_k a power of 2."""
    assert x.is_cuda and x.ndim == 2 and x.dtype == torch.int32
    n_rows, n_elements = x.shape
    N = triton.next_power_of_2(n_elements)
    assert N % stop_k == 0, "padded width must be a multiple of stop_k"

    if stop_k == 1:
        # Sorting a group of size 1 is a no-op — also sidesteps a Triton
        # compiler crash on the degenerate zero-iteration loop this case
        # produces (confirmed on Triton 3.6.0).
        out = torch.full((n_rows, N), -1073741824, dtype=torch.int32, device=x.device)
        out[:, :n_elements] = x
        return out

    x = x.contiguous()
    scratch = torch.empty((n_rows, N), dtype=torch.int32, device=x.device)
    out = torch.empty((n_rows, N), dtype=torch.int32, device=x.device)

    _bitonic_sort_bounded_kernel[(n_rows,)](
        x, scratch, out,
        x.stride(0), scratch.stride(0), out.stride(0),
        n_elements,
        N=N,
        STOP_K=stop_k,
    )
    return out
