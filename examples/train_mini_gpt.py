"""
Mini nanoGPT-style training script wiring in fused_cross_entropy,
FusedRMSNorm, and fused_swiglu, with a --mode fused|native toggle for
real end-to-end tokens/sec + peak-VRAM comparison on an actual model.

Run:
    python examples/train_mini_gpt.py --mode native  --steps 50
    python examples/train_mini_gpt.py --mode fused   --steps 50
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusedkernels.cross_entropy import fused_cross_entropy
from fusedkernels.rmsnorm import FusedRMSNorm
from fusedkernels.swiglu import fused_swiglu


class RefRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, fused: bool):
        super().__init__()
        self.fused = fused
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        h = fused_swiglu(gate, up) if self.fused else F.silu(gate) * up
        return self.w_down(h)


class Block(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, fused: bool):
        super().__init__()
        Norm = FusedRMSNorm if fused else RefRMSNorm
        self.norm1 = Norm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = Norm(dim)
        self.mlp = MLP(dim, hidden_dim, fused)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, hidden_dim, max_seq_len, fused: bool):
        super().__init__()
        self.fused = fused
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, hidden_dim, fused) for _ in range(n_layers)])
        Norm = FusedRMSNorm if fused else RefRMSNorm
        self.norm_f = Norm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, targets):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        if self.fused:
            loss = fused_cross_entropy(logits_flat, targets_flat)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat)
        return loss


def get_batch(batch_size, seq_len, vocab_size, device):
    idx = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    return idx[:, :-1], idx[:, 1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["native", "fused"], required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=2752)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    fused = args.mode == "fused"
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]

    model = MiniGPT(
        vocab_size=args.vocab_size, dim=args.dim, n_layers=args.n_layers,
        n_heads=args.n_heads, hidden_dim=args.hidden_dim, max_seq_len=args.seq_len,
        fused=fused,
    ).to(device=device, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"mode={args.mode}  params={n_params/1e6:.1f}M  dim={args.dim}  layers={args.n_layers}  "
          f"seq_len={args.seq_len}  batch_size={args.batch_size}  dtype={args.dtype}")

    for _ in range(args.warmup):
        x, y = get_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    import time
    start = time.perf_counter()
    total_tokens = 0
    for step in range(args.steps):
        x, y = get_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_tokens += args.batch_size * args.seq_len
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_per_sec = total_tokens / elapsed

    print(f"\n=== RESULTS ({args.mode}) ===")
    print(f"  steps: {args.steps}   time: {elapsed:.2f}s")
    print(f"  tokens/sec: {tokens_per_sec:,.0f}")
    print(f"  peak VRAM: {peak_mem_gb:.2f} GB")
    print(f"  final loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
