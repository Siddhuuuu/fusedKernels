"""
Benchmark fused kernels vs native PyTorch: latency (fwd+bwd) and peak
memory. Run on a CUDA GPU:

    python benchmarks/bench_all.py
"""

import time
import torch
import torch.nn.functional as F

from fusedkernels.cross_entropy import fused_cross_entropy
from fusedkernels.rmsnorm import FusedRMSNorm
from fusedkernels.swiglu import fused_swiglu


def _bench(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return elapsed * 1000, peak_mem_mb


def bench_cross_entropy(N=8192, V=128256):
    print(f"\n=== Cross-Entropy (N={N} tokens, V={V} vocab) ===")

    def native_step(logits, targets):
        logits = logits.clone().requires_grad_(True)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

    def fused_step(logits, targets):
        logits = logits.clone().requires_grad_(True)
        loss = fused_cross_entropy(logits, targets)
        loss.backward()

    logits = torch.randn(N, V, device="cuda", dtype=torch.float16)
    targets = torch.randint(0, V, (N,), device="cuda")

    t_native, m_native = _bench(native_step, logits, targets)
    t_fused, m_fused = _bench(fused_step, logits, targets)

    print(f"  native:  {t_native:7.2f} ms/iter   peak mem: {m_native:8.1f} MB")
    print(f"  fused:   {t_fused:7.2f} ms/iter   peak mem: {m_fused:8.1f} MB")
    print(f"  speedup: {t_native / t_fused:.2f}x   mem reduction: {(1 - m_fused / m_native) * 100:.1f}%")


def bench_rmsnorm(B=4, T=4096, D=4096):
    print(f"\n=== RMSNorm (B={B}, T={T}, D={D}) ===")

    class RefRMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim, device="cuda"))
            self.eps = eps

        def forward(self, x):
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return x * self.weight

    ref_norm = RefRMSNorm(D).cuda()
    fused_norm = FusedRMSNorm(D).cuda()

    def native_step(x):
        x = x.clone().requires_grad_(True)
        y = ref_norm(x)
        y.backward(torch.ones_like(y))

    def fused_step(x):
        x = x.clone().requires_grad_(True)
        y = fused_norm(x)
        y.backward(torch.ones_like(y))

    x = torch.randn(B, T, D, device="cuda", dtype=torch.float16)
    t_native, m_native = _bench(native_step, x)
    t_fused, m_fused = _bench(fused_step, x)

    print(f"  native:  {t_native:7.2f} ms/iter   peak mem: {m_native:8.1f} MB")
    print(f"  fused:   {t_fused:7.2f} ms/iter   peak mem: {m_fused:8.1f} MB")
    print(f"  speedup: {t_native / t_fused:.2f}x   mem reduction: {(1 - m_fused / m_native) * 100:.1f}%")


def bench_swiglu(N=8192, D=14336):
    print(f"\n=== SwiGLU (N={N} tokens, D_ff={D}) ===")

    def native_step(gate, up):
        gate = gate.clone().requires_grad_(True)
        up = up.clone().requires_grad_(True)
        out = F.silu(gate) * up
        out.backward(torch.ones_like(out))

    def fused_step(gate, up):
        gate = gate.clone().requires_grad_(True)
        up = up.clone().requires_grad_(True)
        out = fused_swiglu(gate, up)
        out.backward(torch.ones_like(out))

    gate = torch.randn(N, D, device="cuda", dtype=torch.float16)
    up = torch.randn(N, D, device="cuda", dtype=torch.float16)

    t_native, m_native = _bench(native_step, gate, up)
    t_fused, m_fused = _bench(fused_step, gate, up)

    print(f"  native:  {t_native:7.2f} ms/iter   peak mem: {m_native:8.1f} MB")
    print(f"  fused:   {t_fused:7.2f} ms/iter   peak mem: {m_fused:8.1f} MB")
    print(f"  speedup: {t_native / t_fused:.2f}x   mem reduction: {(1 - m_fused / m_native) * 100:.1f}%")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    bench_cross_entropy()
    bench_rmsnorm()
    bench_swiglu()
