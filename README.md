# fusedkernels

Fused Triton kernels for LLM training that reduce **peak GPU memory** and
**step time** versus stock PyTorch — CrossEntropy, RMSNorm, SwiGLU, and a
from-scratch MoE (Mixture-of-Experts) top-k router built around a hand-
verified bitonic sorting network.

Same technique family as [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
and Unsloth's custom kernels. Built as a systems/kernel-engineering deep
dive — see [`DEBUGGING.md`](DEBUGGING.md) for a real, worked debugging
case study (a non-deterministic race condition, found and fixed).

## What's included

| Kernel | Replaces | Measured result |
|---|---|---|
| `fused_cross_entropy` | `F.cross_entropy` | 2.77-3.06x faster, 20% less peak memory |
| `FusedRMSNorm` | `LlamaRMSNorm` / `nn.RMSNorm` | 3.6-3.89x faster, 39.6% less memory |
| `fused_swiglu` | `F.silu(gate) * up` | 1.5-1.8x faster, 11% less memory |
| `fused_moe_route_v2` | `softmax` + `topk` + renormalize | 1.05-1.40x faster than native `topk` across realistic MoE configs |
| `fused_moe_route_v4` *(experimental)* | same, via truncated bitonic selection | see [Design notes](#design-notes-the-moe-router-story) |
| `FusedMoEMLPSorted` | naive masked-dispatch MoE layer | 1.06-1.32x end-to-end layer speedup |

All numbers measured on an NVIDIA T4 (cloud) and cross-checked on an
RTX 4050 laptop GPU — see [Benchmarking notes](#benchmarking-notes) for
why both matter.

## Install

```bash
pip install torch triton pytest
pip install -e .
```

Requires a CUDA GPU.

## Usage

```python
from fusedkernels import fused_cross_entropy, FusedRMSNorm, fused_swiglu, fused_moe_route_v2, FusedMoEMLPSorted

# Cross-entropy
loss = fused_cross_entropy(logits, targets, ignore_index=-100)

# RMSNorm — drop into any model definition
norm = FusedRMSNorm(hidden_size=4096).cuda()
y = norm(x)

# SwiGLU MLP
h = fused_swiglu(gate, up)

# MoE router
topk_weights, topk_idx = fused_moe_route_v2(router_logits, k=2)

# Full MoE MLP block (sorted dispatch — the fast version)
moe = FusedMoEMLPSorted(dim=1024, hidden_dim=2048, n_experts=8, k=2).cuda()
out = moe(x)
```

## Verify correctness

```bash
pytest tests/ -v
```

Every kernel is checked against the PyTorch reference for both forward
output and backward gradients. The MoE router tests additionally cover
non-power-of-2 K, expert imbalance, extreme logit magnitudes, and a
dedicated **determinism regression test** at the exact scale that once
exposed a real bug (see `DEBUGGING.md`).

## Measure the ROI

```bash
python benchmarks/bench_all.py               # CrossEntropy/RMSNorm/SwiGLU, isolated ops
python examples/moe_router_sweep.py           # router: native vs v1 vs v2 vs v4, across (E, K)
python examples/moe_router_sweep_n.py         # router: does the win change with batch size?
python examples/bench_moe_layer_compare.py    # full MoE layer: naive vs masked-dispatch vs sorted-dispatch
python examples/compare.py --trials 3         # full mini-GPT training loop, fused vs native
```

Numbers depend on GPU architecture, dtype, and shape — run these on your
own hardware rather than trusting numbers from elsewhere. `moe_router_sweep.py`
and `moe_router_sweep_n.py` save results to CSV for further analysis.

## Design notes: the MoE router story

This project has three router implementations, each fixing a real,
measured problem in the previous one — worth understanding in order:

**v1 (`moe_routing.py`) — sequential K-loop.** Selects top-K by K
sequential "find max, mask it, repeat" passes. Simple, but has an O(K)
serial critical path: measured **0.18x-0.40x vs native PyTorch at K=32**
(i.e. 2.5-5x *slower*). Kept in the repo specifically as the baseline
this whole project is measured against.

**v2 (`moe_routing_v2.py`) — the recommended router.** Fuses softmax +
selection + renormalization into one kernel using `tl.sort` (Triton's
own bitonic sorting network), plus a vectorized 2D-tile extraction step
(an earlier version used a naive O(K×E) extraction that reintroduced the
K-scaling problem — see file history). Also fixes a real precision bug:
ranking by post-softmax probability breaks down at large expert counts
because probabilities compress into a tiny range; ranking by the
(monotonic-transformed) logit instead fixes it. **1.05-1.40x faster than
native PyTorch** across realistic (E, K) configs.

**v4 (`moe_routing_v4.py`) — experimental.** Inspired by the technique
FAISS uses for GPU top-k selection (Johnson et al., 2017): instead of
sorting *all* E experts, locally sort small groups (`_bitonic_primitive.py`)
and repeatedly merge them (`_bitonic_merge_primitive.py`), discarding
non-candidates early so later stages only ever touch K-scale data. Targets
O(E log²K) work instead of v2's O(E log²E) — meaningful when K << E, the
realistic MoE regime. **Status:** correct and deterministic for the
tested range after a real bug was found and fixed in the merge kernel —
see `DEBUGGING.md` for the full investigation. Composed via Python
orchestration (verified kernels called in sequence) rather than one
monolithic kernel, trading a little launch overhead for much higher
confidence than hand-rolled multi-level bookkeeping in a single kernel.

## Benchmarking notes

Early runs on a Windows/WSL2 laptop showed inflated speedup numbers
compared to a clean cloud GPU (Kaggle T4) — WSL2's translation layer adds
real overhead to Python/kernel-launch code, which showed up as an
exaggerated "fused beats native" gap. **The numbers in this README are
from the clean cloud runs.** The WSL discrepancy is itself a useful,
honest finding: it's why this repo's benchmark scripts print raw numbers
for you to reproduce rather than hardcoding claims — always benchmark on
your own target hardware.

## Repo structure

```
fusedkernels/
├── fusedkernels/
│   ├── cross_entropy.py, rmsnorm.py, swiglu.py     # elementwise/norm kernels
│   ├── moe_routing.py                               # v1 baseline (naive, kept for comparison)
│   ├── moe_routing_v2.py                            # v2 — main recommended router
│   ├── moe_routing_v4.py                            # v4 — experimental truncated bitonic router
│   ├── _bitonic_primitive.py                        # verified bitonic sort primitives (used by v4)
│   ├── _bitonic_merge_primitive.py                  # verified merge primitive (used by v4)
│   ├── moe_layer.py                                 # naive MoE MLP (masked dispatch, baseline)
│   └── moe_layer_sorted.py                          # sorted-dispatch MoE MLP (the fast version)
├── tests/            # correctness + regression tests for every kernel above
├── benchmarks/        # CrossEntropy/RMSNorm/SwiGLU isolated benchmarks
├── examples/          # router sweeps, layer comparison, end-to-end mini-GPT training
├── DEBUGGING.md        # case study: finding and fixing a real GPU race condition
└── README.md
```

## What this is (and isn't)

This is a personal systems-engineering / kernel-engineering learning
project, not a production library. The *techniques* used here (fused
kernels, bitonic-sort-based selection) are established in the field
(Liger Kernel, FAISS's GPU top-k) — what's original here is the
implementation, verification, and debugging process, documented honestly
including the parts that didn't work on the first try.
