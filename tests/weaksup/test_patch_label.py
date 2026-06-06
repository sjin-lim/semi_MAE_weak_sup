# Unit tests for mask_to_patch_labels / patch_labels_batched.
import pytest

torch = pytest.importorskip("torch")

from dinov3.data.labeled.patch_label import (  # noqa: E402
    mask_to_patch_labels,
    patch_labels_batched,
)


def test_grid_size_and_count():
    ps = 16
    mask = torch.zeros(448, 448, dtype=torch.long)
    labels = mask_to_patch_labels(mask, patch_size=ps)
    n = (448 // ps) ** 2
    assert labels.shape == (n,)  # 28*28 = 784
    assert labels.numel() == 784


def test_pure_blocks_majority():
    ps = 16
    mask = torch.zeros(32, 32, dtype=torch.long)
    # top-left patch all class 1, top-right all class 2,
    # bottom-left class 3, bottom-right class 0
    mask[0:16, 0:16] = 1
    mask[0:16, 16:32] = 2
    mask[16:32, 0:16] = 3
    labels = mask_to_patch_labels(mask, patch_size=ps)  # 2x2 grid → 4 labels
    # flatten order is row-major over (nH, nW)
    assert labels.tolist() == [1, 2, 3, 0]


def test_majority_with_mixed_patch():
    ps = 4
    mask = torch.zeros(4, 4, dtype=torch.long)
    mask[:] = 5
    mask[0, 0] = 7  # 1 of 16 pixels is class 7 → majority still 5
    labels = mask_to_patch_labels(mask, patch_size=ps)
    assert labels.tolist() == [5]


def test_purity_threshold_assigns_ignore():
    ps = 4
    mask = torch.zeros(4, 4, dtype=torch.long)
    # 10 pixels class 1, 6 pixels class 2 → majority 1, purity 10/16 = 0.625
    flat = torch.tensor([1] * 10 + [2] * 6, dtype=torch.long)
    mask = flat.reshape(4, 4)
    # purity 0.625 < 0.7 → ignore
    labels = mask_to_patch_labels(mask, patch_size=ps, min_purity=0.7, ignore_label=-1)
    assert labels.tolist() == [-1]
    # purity 0.625 >= 0.5 → keep majority (1)
    labels2 = mask_to_patch_labels(mask, patch_size=ps, min_purity=0.5, ignore_label=-1)
    assert labels2.tolist() == [1]


def test_purity_off_by_default():
    ps = 4
    flat = torch.tensor([1] * 9 + [2] * 7, dtype=torch.long)
    mask = flat.reshape(4, 4)
    labels = mask_to_patch_labels(mask, patch_size=ps)  # min_purity=0.0
    assert labels.tolist() == [1]


def test_remainder_pixels_dropped():
    ps = 16
    mask = torch.zeros(40, 40, dtype=torch.long)  # 40//16 = 2 → 32x32 used
    labels = mask_to_patch_labels(mask, patch_size=ps)
    assert labels.shape == (4,)  # 2x2, last 8 px row/col ignored


def test_non_2d_raises():
    with pytest.raises(ValueError):
        mask_to_patch_labels(torch.zeros(3, 16, 16, dtype=torch.long))


def test_batched_matches_single():
    ps = 16
    m0 = torch.zeros(32, 32, dtype=torch.long)
    m0[0:16, 0:16] = 1
    m1 = torch.full((32, 32), 2, dtype=torch.long)
    batched = patch_labels_batched(torch.stack([m0, m1]), patch_size=ps)
    assert batched.shape == (2, 4)
    assert torch.equal(batched[0], mask_to_patch_labels(m0, ps))
    assert torch.equal(batched[1], mask_to_patch_labels(m1, ps))


def test_batched_non_3d_raises():
    with pytest.raises(ValueError):
        patch_labels_batched(torch.zeros(32, 32, dtype=torch.long))
