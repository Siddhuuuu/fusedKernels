"""Isolated verification of the hand-written bitonic sort primitives against torch.sort ground truth."""

import pytest
import torch

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA GPU")


@pytest.mark.parametrize("n_elements", [8, 16, 32, 64, 128, 256])
def test_bitonic_matches_torch_sort_power_of_2(n_elements):
    from fusedkernels._bitonic_primitive import bitonic_sort_ascending

    torch.manual_seed(0)
    n_rows = 64
    x = torch.randint(-1000, 1000, (n_rows, n_elements), dtype=torch.int32, device="cuda")

    result = bitonic_sort_ascending(x)
    expected, _ = torch.sort(x, dim=-1, descending=False)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("n_elements", [5, 12, 37, 100, 200])
def test_bitonic_matches_torch_sort_non_power_of_2(n_elements):
    from fusedkernels._bitonic_primitive import bitonic_sort_ascending

    torch.manual_seed(1)
    n_rows = 32
    x = torch.randint(-1000, 1000, (n_rows, n_elements), dtype=torch.int32, device="cuda")

    result = bitonic_sort_ascending(x)
    expected, _ = torch.sort(x, dim=-1, descending=False)
    assert torch.equal(result, expected)


def test_bitonic_handles_duplicate_values():
    from fusedkernels._bitonic_primitive import bitonic_sort_ascending

    n_rows, n_elements = 16, 64
    x = torch.randint(0, 5, (n_rows, n_elements), dtype=torch.int32, device="cuda")

    result = bitonic_sort_ascending(x)
    expected, _ = torch.sort(x, dim=-1, descending=False)
    assert torch.equal(result, expected)


def test_bitonic_single_row_matches():
    from fusedkernels._bitonic_primitive import bitonic_sort_ascending

    x = torch.tensor([[5, 3, 8, 1, 9, 2, 7, 4]], dtype=torch.int32, device="cuda")
    result = bitonic_sort_ascending(x)
    expected = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9]], dtype=torch.int32, device="cuda")
    assert torch.equal(result, expected)
