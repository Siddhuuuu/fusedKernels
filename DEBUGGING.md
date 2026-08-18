# Debugging case study: a non-deterministic race condition in a hand-written GPU kernel

This documents a real bug found while building `moe_routing_v4.py` (the
experimental truncated-bitonic MoE router), from first symptom to fix.
Kept as a written case study rather than a pile of scratch debugging
scripts — this is the process, distilled.

## Symptom

`fused_moe_route_v4` passed correctness tests at small scale and moderate
K, but at K=32 (selecting a large fraction of experts), a small
percentage of rows (2-4%) selected a different set of experts than exact
`torch.topk` — and the *count* of wrong rows varied between otherwise
identical runs.

## Investigation

**1. Ruled out floating-point precision.** Hypothesis: `tl.exp` (Triton)
and `torch.exp` (PyTorch) might round differently, and that tiny
difference gets amplified by the quantization scale used to pack ranking
keys. Confirmed directly: a targeted A/B test showed `tl.exp` and
`torch.exp` disagree on ~16% of elements by ~1e-7 relative error. Fixed
the router to use one consistent code path for key computation — this
narrowed but did **not** close the gap. Ruled out as the primary cause.

**2. Ruled out generic GPU non-determinism.** Ran the *same* router
(v2, the known-good baseline) 5 times on identical input: 0/1024
mismatches, every time, no variance. This ruled out "GPU floating-point
reductions just aren't reproducible" as an explanation — if that were
true, the known-good router would show it too.

**3. Got real GPU tooling working.** Installed the CUDA toolkit + NVIDIA
Compute Sanitizer natively in WSL2 (not the Windows-side binary, which
can't attach to a WSL Linux process — needed the WSL-specific CUDA repo).
Ran `racecheck`, `synccheck`, and `initcheck` — all three came back
clean. This was itself informative: it ruled out classic shared-memory
hazards and synchronization-primitive misuse as the direct cause.

**4. Isolated the bug via systematic bisection.** Rather than guess,
tested each pipeline stage independently for determinism, at both small
and large scale:

| Stage | Small scale (N=8) | Large scale (N=1024+) |
|---|---|---|
| Key computation | deterministic | deterministic |
| Local bounded sort | deterministic | deterministic |
| **Pairwise merge** | deterministic | **non-deterministic** |
| Full pipeline | deterministic | non-deterministic |

This pinned the bug to one specific kernel (`bitonic_merge_topk`'s
hand-rolled compare-exchange implementation) and showed it needed real
GPU concurrency (~1000+ thread blocks) to manifest at all — explaining
why small-scale unit tests never caught it.

**5. Built a minimal, isolated repro.** Just the merge kernel, at the
scale that triggered it (2048 rows), nothing else:
```
wrong rows vs exact ground truth: 87-90/2048, every run
run-to-run variance: 165-175/2048 rows changed between identical runs
```
Confirmed under `compute-sanitizer racecheck`: **the wrongness rate
jumped roughly 10x** (749-849/2048 wrong) under sanitizer instrumentation
— strong indirect evidence of a timing-sensitive bug, even though
`racecheck`'s shared-memory-focused hazard detection didn't directly
flag it (this kernel uses global memory scratch buffers, not CUDA
`__shared__` memory — a different category than what `racecheck` is
built to catch).

## Root cause (best understanding)

The hand-rolled merge kernel does several store→barrier→load round trips
through a global memory scratch buffer (`tl.debug_barrier()` between
each). The exact failure point was never pinned to a specific line —
`compute-sanitizer`'s available tools don't cover this category of
global-memory-ordering issue — but the evidence (isolated to this one
kernel, requires real concurrency, timing-sensitive under
instrumentation) is consistent with a synchronization assumption that
doesn't hold as strongly as expected at scale.

## Fix

Replaced the hand-rolled compare-exchange merge with Triton's own
`tl.sort` (same underlying bitonic-network algorithm family, implemented
and tested by the Triton team) on the small concatenated array, reusing
the same store-to-scratch extraction pattern already proven correct in
v2. Verified at the exact scale that broke the original:

```
0/2048 wrong rows, exact match to ground truth
5/5 identical trials, zero variance
```

## What this demonstrates

- Forming and testing specific hypotheses (A/B testing `tl.exp` vs
  `torch.exp`) rather than guessing at fixes.
- Using GPU-native diagnostic tooling correctly, including recognizing
  *why* a clean sanitizer result doesn't necessarily mean "no bug" — it
  means "no bug in the category this tool checks."
- Systematic bisection to localize a bug across a multi-stage pipeline.
- A pragmatic engineering call: when a hand-rolled implementation has an
  elusive bug, replacing it with a verified library primitive is often
  the right move over continuing to chase the exact line — especially
  once the fix is confirmed correct and deterministic at the scale that
  matters.

See `tests/test_bitonic_merge_primitive.py::test_merge_deterministic_at_scale`
and `tests/test_moe_routing_v4.py::test_fused_moe_route_v4_deterministic_at_scale`
for the regression tests that lock this fix in place.
