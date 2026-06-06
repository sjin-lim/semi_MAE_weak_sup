# Tests for JointGeometricAugmentation / LabeledImageAugmentation.
#
# Key invariants for weak supervision correctness:
#   - mask follows the SAME geometric crop as the image (sizes match)
#   - mask resampling preserves class ids (nearest, no new values)
#   - LabeledImageAugmentation emits 2 global crops + N local crops with
#     matching global_crop_masks, and the mask→patch grid matches a ViT
#     patch grid of (crop_size / patch_size)^2.
import numpy as np
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

# FusedEMIntensity / tv_tensors live here; skip cleanly if heavy deps missing.
jt = pytest.importorskip("dinov3.data.labeled.joint_transforms")
from dinov3.data.labeled.patch_label import mask_to_patch_labels  # noqa: E402

JointGeometricAugmentation = jt.JointGeometricAugmentation
LabeledImageAugmentation = jt.LabeledImageAugmentation


def _toy_image_mask(size=128, n_classes=3):
    img = Image.fromarray((np.random.rand(size, size, 3) * 255).astype(np.uint8))
    mask = torch.from_numpy(
        np.random.randint(0, n_classes, (size, size)).astype(np.int64)
    )
    return img, mask


def test_joint_geometric_size_sync():
    aug = JointGeometricAugmentation(crop_size=64, crop_scale=(0.5, 1.0))
    img, mask = _toy_image_mask(128, n_classes=4)
    out_img, out_mask = aug(img, mask)
    # mask is a plain tensor after augmentation, 2D, matching crop size
    assert isinstance(out_mask, torch.Tensor)
    assert out_mask.dim() == 2, f"mask must be 2D (H,W), got {tuple(out_mask.shape)}"
    assert out_mask.shape == (64, 64)
    assert out_img.size == (64, 64)  # PIL (W, H)


def test_joint_geometric_preserves_class_ids():
    aug = JointGeometricAugmentation(crop_size=64)
    _, mask = _toy_image_mask(128, n_classes=4)
    img = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype(np.uint8))
    orig_ids = set(mask.unique().tolist())
    for _ in range(5):
        _, out_mask = aug(img, mask)
        new_ids = set(out_mask.unique().tolist())
        assert new_ids.issubset(orig_ids), (
            f"nearest resampling must not invent class ids: {new_ids} ⊄ {orig_ids}"
        )


def test_full_pipeline_structure():
    aug = LabeledImageAugmentation(
        global_crops_size=96,
        local_crops_size=48,
        local_crops_number=6,
        instance_norm=False,  # avoid optional util.PerImageNormalize dependency
    )
    img, mask = _toy_image_mask(160, n_classes=3)
    out = aug(img, mask)

    assert len(out["global_crops"]) == 2
    assert len(out["global_crop_masks"]) == 2
    assert len(out["local_crops"]) == 6

    for g in out["global_crops"]:
        assert g.shape == (3, 96, 96)
    for m in out["global_crop_masks"]:
        assert m.dim() == 2 and m.shape == (96, 96)
    for l in out["local_crops"]:
        assert l.shape == (3, 48, 48)


def test_pipeline_mask_to_patch_labels_length():
    patch_size = 16
    crop = 96  # 96/16 = 6 → 36 patches
    aug = LabeledImageAugmentation(
        global_crops_size=crop,
        local_crops_size=48,
        local_crops_number=2,
        instance_norm=False,
    )
    img, mask = _toy_image_mask(160, n_classes=3)
    out = aug(img, mask)
    for m in out["global_crop_masks"]:
        pl = mask_to_patch_labels(m, patch_size=patch_size)
        assert pl.shape == ((crop // patch_size) ** 2,)  # 36


def test_global_crops_are_independent_views():
    # Two global crops should not be identical (independent random crops).
    aug = LabeledImageAugmentation(
        global_crops_size=64,
        local_crops_size=32,
        local_crops_number=1,
        instance_norm=False,
    )
    img, mask = _toy_image_mask(256, n_classes=3)
    out = aug(img, mask)
    g0, g1 = out["global_crops"]
    assert not torch.allclose(g0, g1), "two global views should differ"
