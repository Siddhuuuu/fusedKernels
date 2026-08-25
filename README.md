# fusedkernels

Fused Triton kernels for LLM training - CrossEntropy, RMSNorm, SwiGLU, and
a from-scratch MoE (Mixture-of-Experts) top-k router built around a hand-
verified bitonic sorting network. Reduces peak GPU memory and step time
versus stock PyTorch.

Same technique family as [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
and Unsloth. Built as a kernel-engineering deep dive - see
[`DEBUGGING.md`](DEBUGGING.md) for a real debugging case study: a
non-deterministic race condition, found and fixed.

## Results at a glance

Measured on an NVIDIA T4 (kaggle) and cross-checked on an RTX 4050 laptop.
Numbers vary by hardware/PyTorch version - see [Benchmarking notes](#benchmarking-notes).

| Kernel | Replaces | Typical speedup | Memory saved |
|---|---|---|---|
| `fused_cross_entropy` | `F.cross_entropy` | 1.3-3x | 20% |
| `FusedRMSNorm` | `nn.RMSNorm` | 3.5-3.9x | 40% |
| `fused_swiglu` | `F.silu(gate) * up` | 1.4-1.8x | 11% |
| `fused_moe_route_v2` | `softmax` + `topk` | wins in most configs, up to 1.5x | - |
| `FusedMoEMLPSorted` | naive MoE dispatch | 1.0-1.3x end-to-end | - |
| `fused_moe_route_v4` *(experimental)* | same, via bitonic selection | mixed - see [below](#the-moe-router-story) | - |

## Install

```bash
pip install torch triton pytest
pip install -e .
```
Requires a CUDA GPU.

## Usage

```python
from fusedkernels import fused_cross_entropy, FusedRMSNorm, fused_swiglu, fused_moe_route_v2, FusedMoEMLPSorted

loss = fused_cross_entropy(logits, targets, ignore_index=-100)

norm = FusedRMSNorm(hidden_size=4096).cuda()
y = norm(x)

h = fused_swiglu(gate, up)

topk_weights, topk_idx = fused_moe_route_v2(router_logits, k=2)

moe = FusedMoEMLPSorted(dim=1024, hidden_dim=2048, n_experts=8, k=2).cuda()
out = moe(x)
```

## Verify correctness

```bash
pytest tests/ -v
```

Every kernel is checked against the PyTorch reference for forward output
*and* backward gradients. The MoE router is additionally tested under
expert imbalance, extreme logit values, and a dedicated **determinism
regression test** at the exact scale that once exposed a real bug (see
`DEBUGGING.md`).

## Run the benchmarks yourself

```bash
python benchmarks/bench_all.py               # CrossEntropy / RMSNorm / SwiGLU
python examples/moe_router_sweep.py           # router: native vs v1 vs v2 vs v4
python examples/moe_router_sweep_n.py         # does the router win change with batch size?
python examples/bench_moe_layer_compare.py    # full MoE layer, dispatch strategies
python examples/compare.py --trials 3         # full mini-GPT training loop, fused vs native
```

Numbers depend on your GPU, dtype, and shapes - always benchmark on your
own target hardware rather than trusting numbers from elsewhere.
`moe_router_sweep*.py` save results to CSV.

## The MoE router story

Three router versions, each built to fix a specific, measured problem in
the last one.

**v1 - sequential loop.** Picks the top-K experts with K sequential
"find max, mask, repeat" passes. Simple, but slow at high K: **0.19-0.4x
vs native PyTorch at K=32** (i.e. 2.5-5x *slower*). Kept in the repo as
the baseline everything else is measured against.

**v2 - the one to actually use.** Fuses softmax + selection +
renormalization into one kernel using `tl.sort` (Triton's own bitonic
sort), plus a vectorized extraction step. Also fixes a precision bug:
ranking by post-softmax *probability* breaks down at large expert
counts (probabilities compress into a tiny range); ranking by the
*logit* instead fixes it. **Wins in most realistic configs**, up to 1.5x.

**v4 - experimental, mixed results.** Inspired by how FAISS does GPU
top-k selection: instead of sorting all E experts, sort small groups
and merge them, discarding non-candidates early. In theory this should
need less work than v2's full sort. In practice, measured across three
separate runs (two GPUs, two environments): **v4 underperforms v2 except
at high expert counts (~128), where it starts winning.** Best guess:
v4 launches more kernels per call than v2's one, and that overhead
outweighs the algorithmic savings until E is large enough. Correct and
deterministic everywhere it's tested (a real bug here was found and
fixed - see `DEBUGGING.md`) - just not consistently faster yet. Kept in
the repo as an honest, working experiment, not a recommendation.

## Benchmarking notes

A few things worth knowing before trusting any single number:

- **WSL2 vs cloud GPUs:** early laptop runs (Windows + WSL2) showed
  inflated "fused beats native" numbers - WSL2's translation layer adds
  real overhead to kernel launches. Numbers here are cross-checked on a
  clean cloud GPU (Kaggle T4) for this reason.
- **PyTorch version matters:** native `F.cross_entropy` got noticeably
  faster between PyTorch 2.6 and 2.10 - the fused kernel's advantage
  narrows as PyTorch's own ops improve. This is expected and fine; it's
  why this repo prints raw numbers instead of hardcoding claims.
- **Shared cloud GPUs can be noisy:** one Kaggle run showed the whole
  training loop ~5x slower than a laptop RTX 4050 for both native *and*
  fused - almost certainly GPU contention on a shared instance, not a
  real result. If a number looks like an outlier, rerun it.

## Repo structure

```
fusedkernels/
├── fusedkernels/
│   ├── cross_entropy.py, rmsnorm.py, swiglu.py   # elementwise/norm kernels
│   ├── moe_routing.py                             # v1 - naive baseline
│   ├── moe_routing_v2.py                          # v2 - recommended router
│   ├── moe_routing_v4.py                          # v4 - experimental
│   ├── _bitonic_primitive.py                      # verified sort primitives (used by v4)
│   ├── _bitonic_merge_primitive.py                # verified merge primitive (used by v4)
│   ├── moe_layer.py                               # naive MoE MLP (baseline)
│   └── moe_layer_sorted.py                        # sorted-dispatch MoE MLP (fast)
├── tests/          # correctness + regression tests
├── benchmarks/     # isolated kernel benchmarks
├── examples/       # sweeps, layer comparison, end-to-end training
├── DEBUGGING.md    # case study: a real GPU race condition, found and fixed
└── README.md
```

## What this is (and isn't)

A personal kernel-engineering learning project, not a production
library. The techniques here (fused kernels, bitonic-sort-based
selection) are established in the field - what's original is the
implementation, verification, and debugging, documented honestly
including the parts that didn't work the first time.
