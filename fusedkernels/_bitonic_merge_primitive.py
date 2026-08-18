"""
Merge primitive: combines two already-ascending-sorted top-KP lists into
one top-KP list (the union's top-KP values, ascending sorted). Used as
phase 2 of moe_routing_v4.py's truncated selection network.

IMPLEMENTATION NOTE: uses Triton's own tl.sort rather than a hand-rolled
compare-exchange merge network. An earlier hand-rolled version passed
small-scale correctness tests but showed non-deterministic wrong results
at real GPU concurrency scale (~1000+ rows) — confirmed via direct repro
and isolation testing (see DEBUGGING.md for the full investigation). The
wrongness rate changed by ~10x under compute-sanitizer instrumentation, a
strong signal of a timing-sensitive bug that shared-memory-focused
race detection isn't built to catch (this kernel uses global memory
scratch, not __shared__ memory). Replacing the hand-rolled merge with
tl.sort fully resolved it: 0 wrong results, deterministic across repeated
trials at the exact scale that broke the original.

Trade-off: does a full tl.sort of the 2*KP concatenated array instead of
a pure O(log(2KP))-depth merge network — slightly more work than the
original design intended, but KP stays small (32-64 typically), so the
difference is minor. This keeps the actual point of the design (never
fully sort all E experts — only ever sort/merge small KP-scale chunks)
while eliminating the specific broken kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_merge_topk_kernel(
    a_ptr, b_ptr, out_ptr, scratch_ptr,
    row_stride_a, row_stride_b, row_stride_out, row_stride_scratch,
    KP: tl.constexpr,
    N2: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, N2)

    a_row = a_ptr + row * row_stride_a
    b_row = b_ptr + row * row_stride_b
    scratch_row = scratch_ptr + row * row_stride_scratch

    is_first_half = idx < KP
    a_part_idx = idx
    b_part_idx = idx - KP

    a_vals = tl.load(a_row + a_part_idx, mask=is_first_half, other=-1073741824)
    b_vals = tl.load(b_row + b_part_idx, mask=~is_first_half, other=-1073741824)
    combined = tl.where(is_first_half, a_vals, b_vals)

    # ONE sort call — Triton's own proven-robust primitive
    sorted_combined = tl.sort(combined, descending=False)

    tl.store(scratch_row + idx, sorted_combined)
    tl.debug_barrier()

    out_positions = tl.arange(0, KP)
    top_vals = tl.load(scratch_row + (out_positions + KP))
    out_row = out_ptr + row * row_stride_out
    tl.store(out_row + out_positions, top_vals)


def bitonic_merge_topk(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: [n_rows, KP] int32, each already ascending-sorted, KP a power
    of 2. Returns [n_rows, KP]: the top-KP values from the union of each
    row's a and b, ascending sorted."""
    assert a.is_cuda and b.is_cuda and a.shape == b.shape and a.dtype == torch.int32
    n_rows, KP = a.shape
    assert (KP & (KP - 1)) == 0, "KP must be a power of 2"
    N2 = 2 * KP

    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty((n_rows, KP), dtype=torch.int32, device=a.device)
    scratch = torch.empty((n_rows, N2), dtype=torch.int32, device=a.device)

    _bitonic_merge_topk_kernel[(n_rows,)](
        a, b, out, scratch,
        a.stride(0), b.stride(0), out.stride(0), scratch.stride(0),
        KP=KP,
        N2=N2,
    )
    return out
