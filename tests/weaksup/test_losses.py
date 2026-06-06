# Unit tests for per_patch_pairwise_loss / batch_weak_loss / compute_pair_sim_stats.
#
# Covers the core claims from HANDOFF_CONTEXT.md §4.3 / §6:
#   - random (dissimilar) features  → loss ≈ 0.001 (negligible)
#   - artificially similar features → loss large (0.3+)
#   - loss bounded in [0, 1]
#   - background class excluded
#   - min_patches_per_class gating
#   - gradient flows to features (push-only direction)
#   - self-curriculum: monotonically lower sim → lower loss
import math

import pytest

torch = pytest.importorskip("torch")

from dinov3.train.weaksup.losses import (  # noqa: E402
    per_patch_pairwise_loss,
    batch_weak_loss,
    compute_pair_sim_stats,
)


def _make_labels(counts):
    """counts: dict {class_id: n_patches} -> concatenated label tensor."""
    parts = [torch.full((n,), c, dtype=torch.long) for c, n in counts.items()]
    return torch.cat(parts)


def test_random_features_negligible_loss():
    torch.manual_seed(0)
    N, D = 100, 768
    feats = torch.randn(N, D)
    labels = _make_labels({0: 30, 1: 35, 2: 35})  # 0 = background

    loss = per_patch_pairwise_loss(feats, labels, T=8.0)
    # Random high-dim unit vectors → cross-class cosine ~0 → exp(8*~0)-1 ~0
    assert loss.item() < 0.01, f"expected tiny loss for random features, got {loss.item()}"


def test_similar_features_large_loss():
    torch.manual_seed(0)
    D = 768
    base1 = torch.randn(D)
    base2 = base1 + 0.05 * torch.randn(D)  # class 2 ≈ class 1 (confused pair)
    feats = torch.cat([
        torch.randn(30, D),                          # bg
        base1.unsqueeze(0) + 0.02 * torch.randn(35, D),  # class 1
        base2.unsqueeze(0) + 0.02 * torch.randn(35, D),  # class 2
    ])
    labels = _make_labels({0: 30, 1: 35, 2: 35})

    loss = per_patch_pairwise_loss(feats, labels, T=8.0)
    assert loss.item() > 0.3, f"expected large loss for confused pair, got {loss.item()}"


def test_loss_bounded_unit_interval():
    torch.manual_seed(1)
    D = 64
    # Identical features across two classes → max possible sim = 1.0
    v = torch.randn(D)
    feats = torch.cat([v.unsqueeze(0).repeat(20, 1), v.unsqueeze(0).repeat(20, 1)])
    labels = _make_labels({1: 20, 2: 20})
    loss = per_patch_pairwise_loss(feats, labels, T=8.0)
    assert 0.0 <= loss.item() <= 1.0 + 1e-5, f"loss out of [0,1]: {loss.item()}"
    # sim == 1 everywhere → normalized exp == (exp(T)-1)/(exp(T)-1) == 1
    assert loss.item() == pytest.approx(1.0, abs=1e-3)


def test_background_excluded():
    torch.manual_seed(2)
    D = 64
    v = torch.randn(D)
    # bg (class 0) identical to class 1, but bg must be skipped → no pair
    feats = torch.cat([v.unsqueeze(0).repeat(20, 1), v.unsqueeze(0).repeat(20, 1)])
    labels = _make_labels({0: 20, 1: 20})
    loss = per_patch_pairwise_loss(feats, labels, T=8.0, skip_background=True)
    assert loss.item() == 0.0, "only bg+1 class present → no non-bg pair → zero"


def test_min_patches_gating():
    torch.manual_seed(3)
    D = 64
    v = torch.randn(D)
    feats = torch.cat([v.unsqueeze(0).repeat(10, 1), v.unsqueeze(0).repeat(2, 1)])
    labels = _make_labels({1: 10, 2: 2})  # class 2 below min_patches=4
    loss = per_patch_pairwise_loss(feats, labels, T=8.0, min_patches_per_class=4)
    assert loss.item() == 0.0, "class 2 too small → dropped → single class → zero"


def test_single_class_zero():
    feats = torch.randn(20, 32)
    labels = torch.ones(20, dtype=torch.long)
    assert per_patch_pairwise_loss(feats, labels).item() == 0.0


def test_ignore_label_dropped():
    torch.manual_seed(4)
    D = 64
    v = torch.randn(D)
    feats = torch.cat([
        v.unsqueeze(0).repeat(20, 1),   # class 1
        v.unsqueeze(0).repeat(20, 1),   # class 2 (identical → would be loss 1)
        torch.randn(20, D),             # ignore (-1)
    ])
    labels = _make_labels({1: 20, 2: 20})
    labels = torch.cat([labels, torch.full((20,), -1, dtype=torch.long)])
    loss = per_patch_pairwise_loss(feats, labels, T=8.0)
    # ignore patches must not create pairs; classes 1 & 2 still do → ~1.0
    assert loss.item() == pytest.approx(1.0, abs=1e-3)


def test_gradient_flows_and_pushes_apart():
    torch.manual_seed(5)
    D = 32
    v = torch.randn(D)
    feats = torch.cat([
        v.unsqueeze(0).repeat(8, 1) + 0.01 * torch.randn(8, D),
        v.unsqueeze(0).repeat(8, 1) + 0.01 * torch.randn(8, D),
    ]).requires_grad_(True)
    labels = _make_labels({1: 8, 2: 8})
    loss = per_patch_pairwise_loss(feats, labels, T=8.0)
    loss.backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
    assert feats.grad.abs().sum() > 0, "push force should produce nonzero gradient"


def test_self_curriculum_monotonic():
    # As cross-class sim decreases, loss must decrease (self-curriculum claim).
    torch.manual_seed(6)
    D = 128
    losses = []
    for noise in [0.0, 0.3, 0.7, 1.5]:
        v = torch.randn(D)
        feats = torch.cat([
            v.unsqueeze(0).repeat(20, 1),
            v.unsqueeze(0).repeat(20, 1) + noise * torch.randn(20, D),
        ])
        labels = _make_labels({1: 20, 2: 20})
        losses.append(per_patch_pairwise_loss(feats, labels, T=8.0).item())
    for a, b in zip(losses, losses[1:]):
        assert a >= b - 1e-4, f"loss should be non-increasing as sim drops: {losses}"


def test_T_sharpness_effect():
    # Higher T → only very-high-sim region penalized → lower loss at moderate sim.
    torch.manual_seed(7)
    D = 128
    v = torch.randn(D)
    feats = torch.cat([
        v.unsqueeze(0).repeat(20, 1),
        v.unsqueeze(0).repeat(20, 1) + 0.6 * torch.randn(20, D),  # moderate sim
    ])
    labels = _make_labels({1: 20, 2: 20})
    loss_low_T = per_patch_pairwise_loss(feats, labels, T=4.0).item()
    loss_high_T = per_patch_pairwise_loss(feats, labels, T=12.0).item()
    assert loss_high_T < loss_low_T, "higher T concentrates force on high-sim only"


def test_batch_weak_loss_aggregation():
    torch.manual_seed(8)
    D = 64
    v = torch.randn(D)
    # image 0: confused pair → nonzero; image 1: random → ~zero
    img0 = torch.cat([v.unsqueeze(0).repeat(10, 1), v.unsqueeze(0).repeat(10, 1)])
    lbl0 = _make_labels({1: 10, 2: 10})
    img1 = torch.randn(20, D)
    lbl1 = _make_labels({1: 10, 2: 10})

    out = batch_weak_loss([img0, img1], [lbl0, lbl1], T=8.0)
    assert out.item() > 0.0
    # batch loss averages only contributing images; img0 dominates
    only0 = per_patch_pairwise_loss(img0, lbl0, T=8.0).item()
    assert out.item() == pytest.approx(only0, abs=0.05)


def test_batch_weak_loss_empty():
    out = batch_weak_loss([], [])
    assert out.item() == 0.0


def test_compute_pair_sim_stats():
    torch.manual_seed(9)
    D = 64
    v = torch.randn(D)
    feats = torch.cat([
        torch.randn(20, D),               # bg
        v.unsqueeze(0).repeat(20, 1),     # class 1
        v.unsqueeze(0).repeat(20, 1),     # class 2 == class 1 centroid
    ])
    labels = _make_labels({0: 20, 1: 20, 2: 20})
    stats = compute_pair_sim_stats(feats, labels)
    assert stats["n_pairs"] == 1  # only the 1-2 pair (bg excluded)
    assert stats["pair_sim_max"] == pytest.approx(1.0, abs=1e-3)
    assert stats["hard_pair_ratio"] == pytest.approx(1.0)
