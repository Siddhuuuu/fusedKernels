"""Verification of the (fixed, tl.sort-based) bitonic top-K merge primitive."""

import pytest
import torch

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA GPU")


@pytest.mark.parametrize("KP", [2, 4, 8, 16, 32])
def test_merge_matches_ground_truth(KP):
    from fusedkernels._bitonic_merge_primitive import bitonic_merge_topk

    torch.manual_seed(0)
    n_rows = 128
    a_raw = torch.randint(-1000, 1000, (n_rows, KP), dtype=torch.int32, device="cuda")
    b_raw = torch.randint(-1000, 1000, (n_rows, KP), dtype=torch.int32, device="cuda")

    a_sorted, _ = torch.sort(a_raw, dim=-1)
    b_sorted, _ = torch.sort(b_raw, dim=-1)

    result = bitonic_merge_topk(a_sorted, b_sorted)

    combined = torch.cat([a_sorted, b_sorted], dim=-1)
    expected, _ = torch.sort(combined, dim=-1)
    expected_top = expected[:, KP:]

    assert torch.equal(result, expected_top), f"merge mismatch at KP={KP}"


@pytest.mark.parametrize("n_rows,KP", [(2048, 32), (2048, 64)])
def test_merge_deterministic_at_scale(n_rows, KP):
    """Regression test for the specific bug found and fixed: the earlier
    hand-rolled merge kernel was wrong AND non-deterministic at this
    scale (~4% of rows wrong, varying between runs). This scale is what
    actually appears in the real MoE routing pipeline at large N."""
    from fusedkernels._bitonic_merge_primitive import bitonic_merge_topk

    torch.manual_seed(0)
    a_raw = torch.randint(-1000, 1000, (n_rows, KP), dtype=torch.int32, device="cuda")
    b_raw = torch.randint(-1000, 1000, (n_rows, KP), dtype=torch.int32, device="cuda")
    a_sorted, _ = torch.sort(a_raw, dim=-1)
    b_sorted, _ = torch.sort(b_raw, dim=-1)

    combined = torch.cat([a_sorted, b_sorted], dim=-1)
    expected, _ = torch.sort(combined, dim=-1)
    expected_top = expected[:, KP:]

    first_result = None
    for trial in range(5):
        result = bitonic_merge_topk(a_sorted.clone(), b_sorted.clone())
        assert torch.equal(result, expected_top), \
            f"trial {trial}: wrong vs ground truth at scale n_rows={n_rows}"
        if first_result is None:
            first_result = result.clone()
        else:
            assert torch.equal(result, first_result), \
                f"trial {trial}: non-deterministic at scale n_rows={n_rows}"


def test_merge_handles_duplicates_across_lists():
    from fusedkernels._bitonic_merge_primitive import bitonic_merge_topk

    n_rows, KP = 32, 16
    a_raw = torch.randint(0, 10, (n_rows, KP), dtype=torch.int32, device="cuda")
    b_raw = torch.randint(0, 10, (n_rows, KP), dtype=torch.int32, device="cuda")

    a_sorted, _ = torch.sort(a_raw, dim=-1)
    b_sorted, _ = torch.sort(b_raw, dim=-1)

    result = bitonic_merge_topk(a_sorted, b_sorted)

    combined = torch.cat([a_sorted, b_sorted], dim=-1)
    expected, _ = torch.sort(combined, dim=-1)
    expected_top = expected[:, KP:]

    assert torch.equal(result, expected_top)


def test_merge_hand_checked_example():
    from fusedkernels._bitonic_merge_primitive import bitonic_merge_topk

    a = torch.tensor([[1, 3, 5, 9]], dtype=torch.int32, device="cuda")
    b = torch.tensor([[2, 4, 6, 8]], dtype=torch.int32, device="cuda")
    result = bitonic_merge_topk(a, b)
    expected = torch.tensor([[5, 6, 8, 9]], dtype=torch.int32, device="cuda")
    assert torch.equal(result, expected)
