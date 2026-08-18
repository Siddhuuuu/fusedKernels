"""Verification of bitonic_sort_bounded: independent per-group sorting."""

import pytest
import torch

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA GPU")


@pytest.mark.parametrize("n_groups,group_size", [(2, 2), (4, 2), (2, 4), (4, 4), (8, 4), (4, 8), (2, 16), (2, 32), (4, 32)])
def test_bounded_sort_matches_per_group_torch_sort(n_groups, group_size):
    from fusedkernels._bitonic_primitive import bitonic_sort_bounded

    torch.manual_seed(0)
    n_rows = 32
    n_elements = n_groups * group_size
    x = torch.randint(-1000, 1000, (n_rows, n_elements), dtype=torch.int32, device="cuda")

    result = bitonic_sort_bounded(x, group_size)
    expected = x.view(n_rows, n_groups, group_size).sort(dim=-1)[0].view(n_rows, n_elements)

    assert torch.equal(result[:, :n_elements], expected), \
        f"bounded sort mismatch at n_groups={n_groups}, group_size={group_size}"


def test_bounded_sort_groups_dont_interfere():
    from fusedkernels._bitonic_primitive import bitonic_sort_bounded

    n_rows, group_size = 4, 8
    g1 = torch.randint(-500, 500, (n_rows, group_size), dtype=torch.int32, device="cuda")
    g2 = torch.zeros((n_rows, group_size), dtype=torch.int32, device="cuda")
    g3 = torch.full((n_rows, group_size), 999999, dtype=torch.int32, device="cuda")
    x = torch.cat([g1, g2, g3], dim=-1)

    result = bitonic_sort_bounded(x, group_size)
    expected = x.view(n_rows, 3, group_size).sort(dim=-1)[0].view(n_rows, -1)

    assert torch.equal(result[:, :x.shape[1]], expected)
