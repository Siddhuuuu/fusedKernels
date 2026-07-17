"""
Diagnostic for the (128 experts, k=32) anomaly in bench_moe_layer_compare.py.
Checks peak VRAM usage for that specific config to see if it's close to
this GPU's limit (suggesting allocator thrashing) rather than a real
algorithmic slowdown.

Run:
    python examples/moe_layer_memory_check.py
"""

import torch

from fusedkernels.moe_layer_sorted import FusedMoEMLPSorted


def main():
    assert torch.cuda.is_available()
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {torch.cuda.get_device_name(0)}  (total VRAM: {total_vram_gb:.2f} GB)\n")

    B, T, D, H = 4, 512, 1024, 2048
    n_experts, k = 128, 32

    torch.cuda.reset_peak_memory_stats()
    layer = FusedMoEMLPSorted(D, H, n_experts, k, router_version="v2").cuda().to(torch.bfloat16)
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    for i in range(5):
        out = layer(x)
        out.sum().backward()
        layer.zero_grad(set_to_none=True)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"iter {i}: peak VRAM so far = {peak_gb:.3f} GB  "
              f"({100*peak_gb/total_vram_gb:.1f}% of total)")

    print("\nIf peak VRAM is within ~10-15% of total, that config is likely")
    print("hitting allocator pressure — explains the anomalous slowdown.")


if __name__ == "__main__":
    main()
