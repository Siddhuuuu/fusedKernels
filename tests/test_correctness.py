"""Correctness tests: fused CrossEntropy/RMSNorm/SwiGLU vs native PyTorch."""

import pytest
import torch
import torch.nn.functional as F

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA GPU")


def test_fused_cross_entropy_matches_pytorch():
    from fusedkernels.cross_entropy import fused_cross_entropy

    torch.manual_seed(0)
    N, V = 512, 32000
    logits_ref = torch.randn(N, V, device="cuda", dtype=torch.float32, requires_grad=True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, V, (N,), device="cuda")
    targets[::37] = -100

    loss_ref = F.cross_entropy(logits_ref, targets, ignore_index=-100)
    loss_fused = fused_cross_entropy(logits_fused, targets, ignore_index=-100)
    assert torch.allclose(loss_ref, loss_fused, atol=1e-3, rtol=1e-3)

    loss_ref.backward()
    loss_fused.backward()
    assert torch.allclose(logits_ref.grad, logits_fused.grad, atol=1e-3, rtol=1e-3)


def test_fused_cross_entropy_label_smoothing():
    from fusedkernels.cross_entropy import fused_cross_entropy

    torch.manual_seed(1)
    N, V = 256, 4096
    logits_ref = torch.randn(N, V, device="cuda", requires_grad=True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, V, (N,), device="cuda")

    loss_ref = F.cross_entropy(logits_ref, targets, label_smoothing=0.1)
    loss_fused = fused_cross_entropy(logits_fused, targets, label_smoothing=0.1)
    assert torch.allclose(loss_ref, loss_fused, atol=1e-3, rtol=1e-3)


def test_fused_rmsnorm_matches_pytorch():
    from fusedkernels.rmsnorm import FusedRMSNorm

    class RefRMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return x * self.weight

    torch.manual_seed(2)
    B, T, D = 4, 128, 4096
    x_ref = torch.randn(B, T, D, device="cuda", requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)

    ref_norm = RefRMSNorm(D).cuda()
    fused_norm = FusedRMSNorm(D).cuda()
    fused_norm.weight.data.copy_(ref_norm.weight.data)

    y_ref = ref_norm(x_ref)
    y_fused = fused_norm(x_fused)
    assert torch.allclose(y_ref, y_fused, atol=1e-3, rtol=1e-3)

    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)
    y_fused.backward(grad_out)

    assert torch.allclose(x_ref.grad, x_fused.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ref_norm.weight.grad, fused_norm.weight.grad, atol=1e-2, rtol=1e-2)


def test_fused_swiglu_matches_pytorch():
    from fusedkernels.swiglu import fused_swiglu

    torch.manual_seed(3)
    N, D = 2048, 4096
    gate_ref = torch.randn(N, D, device="cuda", requires_grad=True)
    up_ref = torch.randn(N, D, device="cuda", requires_grad=True)
    gate_fused = gate_ref.detach().clone().requires_grad_(True)
    up_fused = up_ref.detach().clone().requires_grad_(True)

    out_ref = F.silu(gate_ref) * up_ref
    out_fused = fused_swiglu(gate_fused, up_fused)
    assert torch.allclose(out_ref, out_fused, atol=1e-3, rtol=1e-3)

    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_fused.backward(grad_out)

    assert torch.allclose(gate_ref.grad, gate_fused.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(up_ref.grad, up_fused.grad, atol=1e-3, rtol=1e-3)
