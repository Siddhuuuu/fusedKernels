# fusedkernels

Drop-in fused Triton kernels for LLM training that reduce **peak GPU memory**
and **step time** versus stock PyTorch ops. Same technique family as
[Liger Kernel](https://github.com/linkedin/Liger-Kernel) / Unsloth's custom
kernels — usable in any training loop (HF `Trainer`, nanoGPT, custom loops).

## What's included

| Kernel | Replaces | Why it helps |
|---|---|---|
| `fused_cross_entropy` | `F.cross_entropy` | Never materializes the full `[N, vocab]` softmax probability tensor — biggest single memory win for large-vocab LLMs |
| `FusedRMSNorm` | `LlamaRMSNorm` / `nn.RMSNorm` | Fuses 4-5 elementwise ops into one kernel pass (bandwidth-bound op) |
| `fused_swiglu` | `F.silu(gate) * up` | Fuses activation + multiply, skips materializing the intermediate `silu(gate)` tensor |

## Install

```bash
pip install -e .
```

Requires a CUDA GPU, `torch>=2.1`, `triton>=2.1`.

## Usage

```python
from fusedkernels import fused_cross_entropy, FusedRMSNorm, fused_swiglu

# Cross-entropy
loss = fused_cross_entropy(logits, targets, ignore_index=-100)

# RMSNorm — drop into any model definition
norm = FusedRMSNorm(hidden_size=4096).cuda()
y = norm(x)

# SwiGLU MLP
gate = x @ W_gate
up = x @ W_up
h = fused_swiglu(gate, up)
out = h @ W_down
```

## Verify correctness

```bash
pytest tests/test_correctness.py -v
```

Each kernel is checked against the PyTorch reference implementation for
both forward output and backward gradients (`torch.allclose`).

## Measure the ROI on your own hardware

```bash
python benchmarks/bench_all.py
```

Prints native-vs-fused latency (ms/iter) and peak memory (MB) for realistic
LLM-scale shapes (128k vocab, 4096 hidden dim, etc.) — edit the shapes in
`benchmarks/bench_all.py` to match your actual model config. Report the
speedup and memory delta from your own run; numbers depend heavily on GPU
architecture, dtype, and shape, so don't trust numbers from elsewhere.

## End-to-end benchmark (real model, not isolated ops)

`examples/train_mini_gpt.py` is a small nanoGPT-style transformer that
wires in all three fused kernels behind a `--mode fused|native` flag, so
you can compare actual training tokens/sec and peak VRAM on a real model
rather than isolated op benchmarks:

```bash
python examples/train_mini_gpt.py --mode native --steps 50
python examples/train_mini_gpt.py --mode fused  --steps 50
```

Both print `tokens/sec`, `peak VRAM (GB)`, and final loss — diff the two
runs for your headline ROI number. Defaults to a ~small model (dim=1024,
12 layers, 16 heads) with synthetic random tokens (pure throughput/memory
benchmark, no dataset needed). Flags let you match your real target scale:

```bash
python examples/train_mini_gpt.py --mode fused \
  --dim 2048 --n-layers 24 --n-heads 16 --hidden-dim 5632 \
  --batch-size 4 --seq-len 2048 --vocab-size 128256 --dtype bf16 --steps 50
```

Pass `--data path/to/tokens.bin` (uint16 token ids, e.g. from a nanoGPT
`prepare.py`-style pipeline) to train on real tokenized text instead of
random tokens.

## How to plug into a real training loop

- **Cross-entropy**: replace `nn.CrossEntropyLoss()(logits, targets)` or
  `F.cross_entropy(...)` directly with `fused_cross_entropy(...)`.
- **RMSNorm**: swap the norm class in your model definition (e.g. HF Llama
  models use `LlamaRMSNorm` — same math, same `weight`/`eps` interface).
- **SwiGLU**: in the MLP block, replace `F.silu(gate) * up` with
  `fused_swiglu(gate, up)`.

## Notes / next steps

- Kernels are single-block-per-row; extremely large hidden/vocab dims
  (beyond ~64k for RMSNorm, or very tall BLOCK_SIZE for CE) may need
  further tiling — check `benchmarks/` output for your shapes.
- `label_smoothing` is supported in `fused_cross_entropy`.
- No autotuning yet — `BLOCK_SIZE`/`num_warps` are heuristic. A natural
  next step is `triton.autotune` per-GPU-arch tuning for extra speedup.
