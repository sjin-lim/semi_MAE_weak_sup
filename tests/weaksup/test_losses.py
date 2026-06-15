# Unit tests for per_patch_pairwise_loss / batch_weak_loss / compute_pair_sim_stats.
#
# Loss is HINGE ANTI-MERGE:  ℓ_ij = (relu(sim - margin) / (1-margin))^power
#   - cross-class sim <= margin   → contributes exactly 0 (이미 구분된 쌍 방치)
#   - cross-class sim == 1        → contributes 1.0 (fully merged)
#   - bounded [0, 1], self-curriculum (sim 떨어지면 자동 0)
#   - background excluded, min_patches gating, gradient flows on >margin pairs

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


def _two_class_feats(sim, n=10, D=64):
    """두 class 간 cross cosine 을 정확히 `sim` 으로 만드는 feature/label."""
    a = torch.zeros(n, D)
    a[:, 0] = 1.0
    b = torch.zeros(n, D)
    b[:, 0] = sim
    b[:, 1] = (1.0 - sim ** 2) ** 0.5
    feats = torch.cat([a, b])
    labels = _make_labels({1: n, 2: n})
    return feats, labels


def test_random_features_negligible_loss():
    torch.manual_seed(0)
    N, D = 100, 768
    feats = torch.randn(N, D)
    labels = _make_labels({0: 30, 1: 35, 2: 35})  # 0 = background

    loss = per_patch_pairwise_loss(feats, labels, margin=0.85)
    # Random high-dim vectors → cross-class cosine ~0 << margin → hinge 0
    assert loss.item() < 1e-6, f"expected ~0 loss for random features, got {loss.item()}"


def test_similar_features_large_loss():
    torch.manual_seed(0)
    D = 768
    base1 = torch.randn(D)
    base2 = base1 + 0.05 * torch.randn(D)  # class 2 ≈ class 1 (confused pair)
    feats = torch.cat([
        torch.randn(30, D),                              # bg
        base1.unsqueeze(0) + 0.02 * torch.randn(35, D),  # class 1
        base2.unsqueeze(0) + 0.02 * torch.randn(35, D),  # class 2
    ])
    labels = _make_labels({0: 30, 1: 35, 2: 35})

    loss = per_patch_pairwise_loss(feats, labels, margin=0.85)
    assert loss.item() > 0.3, f"expected large loss for merged pair, got {loss.item()}"


def test_loss_bounded_unit_interval():
    torch.manual_seed(1)
    D = 64
    # Identical features across two classes → sim = 1.0 everywhere
    v = torch.randn(D)
    feats = torch.cat([v.unsqueeze(0).repeat(20, 1), v.unsqueeze(0).repeat(20, 1)])
    labels = _make_labels({1: 20, 2: 20})
    loss = per_patch_pairwise_loss(feats, labels, margin=0.85)
    assert 0.0 <= loss.item() <= 1.0 + 1e-5, f"loss out of [0,1]: {loss.item()}"
    # sim==1 → (relu(1-margin)/(1-margin))^2 == 1
    assert loss.item() == pytest.approx(1.0, abs=1e-3)


def test_below_margin_zero_force():
    # 핵심: margin 이하로 이미 구분된 쌍은 force 가 정확히 0.
    feats, labels = _two_class_feats(sim=0.7, n=10)
    assert per_patch_pairwise_loss(feats, labels, margin=0.85).item() == 0.0
    # margin 을 sim 아래로 내리면 다시 force 발생
    assert per_patch_pairwise_loss(feats, labels, margin=0.6).item() > 0.0


def test_margin_higher_means_lower_loss():
    # 같은 (높은) sim 에서 margin 이 클수록 loss 가 작다.
    feats, labels = _two_class_feats(sim=0.95, n=10)
    low_margin = per_patch_pairwise_loss(feats, labels, margin=0.6).item()
    high_margin = per_patch_pairwise_loss(feats, labels, margin=0.85).item()
    assert high_margin < low_margin, (low_margin, high_margin)


def test_background_excluded():
    torch.manual_seed(2)
    D = 64
    v = torch.randn(D)
    # bg (class 0) identical to class 1, but bg must be skipped → no pair
    feats = torch.cat([v.unsqueeze(0).repeat(20, 1), v.unsqueeze(0).repeat(20, 1)])
    labels = _make_labels({0: 20, 1: 20})
    loss = per_patch_pairwise_loss(feats, labels, margin=0.85, skip_background=True)
    assert loss.item() == 0.0, "only bg+1 class present → no non-bg pair → zero"


def test_min_patches_gating():
    torch.manual_seed(3)
    D = 64
    v = torch.randn(D)
    feats = torch.cat([v.unsqueeze(0).repeat(10, 1), v.unsqueeze(0).repeat(2, 1)])
    labels = _make_labels({1: 10, 2: 2})  # class 2 below min_patches=4
    loss = per_patch_pairwise_loss(feats, labels, margin=0.85, min_patches_per_class=4)
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
    loss = per_patch_pairwise_loss(feats, labels, margin=0.85)
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
    labels = _make_labels({1: 8, 2: 8})  # sim ≈ 1 > margin → force on
    loss = per_patch_pairwise_loss(feats, labels, margin=0.85)
    loss.backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
    assert feats.grad.abs().sum() > 0, "merged pair should produce nonzero gradient"


def test_self_curriculum_monotonic():
    # As cross-class sim decreases, loss must be non-increasing (self-curriculum).
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
        losses.append(per_patch_pairwise_loss(feats, labels, margin=0.85).item())
    for a, b in zip(losses, losses[1:]):
        assert a >= b - 1e-4, f"loss should be non-increasing as sim drops: {losses}"


def test_batch_weak_loss_excludes_zero_images():
    # batch_weak_loss averages ONLY images whose per-image loss > 0.
    torch.manual_seed(8)
    D = 64
    v = torch.randn(D)
    # image 0: merged pair → nonzero
    img0 = torch.cat([v.unsqueeze(0).repeat(10, 1), v.unsqueeze(0).repeat(10, 1)])
    lbl0 = _make_labels({1: 10, 2: 10})
    # image 1: single class → per-image loss is exactly 0 → excluded
    img1 = torch.randn(20, D)
    lbl1 = torch.ones(20, dtype=torch.long)

    out = batch_weak_loss([img0, img1], [lbl0, lbl1], margin=0.85)
    only0 = per_patch_pairwise_loss(img0, lbl0, margin=0.85).item()
    assert out.item() == pytest.approx(only0, abs=1e-4), (
        "img1 contributes 0 → must be dropped → batch == img0 loss"
    )


def test_batch_weak_loss_is_plain_mean():
    # Two contributing (merged) images → unweighted mean of per-image losses.
    torch.manual_seed(8)
    D = 64
    v = torch.randn(D)
    w = torch.randn(D)
    img0 = torch.cat([v.unsqueeze(0).repeat(10, 1), v.unsqueeze(0).repeat(10, 1)])
    lbl0 = _make_labels({1: 10, 2: 10})
    img1 = torch.cat([w.unsqueeze(0).repeat(10, 1), w.unsqueeze(0).repeat(10, 1)])
    lbl1 = _make_labels({1: 10, 2: 10})

    l0 = per_patch_pairwise_loss(img0, lbl0, margin=0.85).item()
    l1 = per_patch_pairwise_loss(img1, lbl1, margin=0.85).item()
    assert l0 > 0 and l1 > 0
    expected = (l0 + l1) / 2.0

    out = batch_weak_loss([img0, img1], [lbl0, lbl1], margin=0.85)
    assert out.item() == pytest.approx(expected, abs=1e-5)


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
    stats = compute_pair_sim_stats(feats, labels, margin=0.85)
    assert stats["n_pairs"] == 1  # only the 1-2 pair (bg excluded)
    assert stats["pair_sim_max"] == pytest.approx(1.0, abs=1e-3)
    assert stats["merged_pair_ratio"] == pytest.approx(1.0)
