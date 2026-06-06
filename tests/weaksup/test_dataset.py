# Smoke tests for LabeledEMDataset using a toy folder built on tmp_path.
import json

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from dinov3.data.labeled.dataset import LabeledEMDataset  # noqa: E402


def _build_toy_root(root, datasets, n_per=3, size=64, n_classes=3, with_meta=True):
    """Create labeled_root/<ds>/{images,masks}/img_*.png + optional meta.json."""
    for ds in datasets:
        img_dir = root / ds / "images"
        mask_dir = root / ds / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for i in range(n_per):
            stem = f"img_{i:03d}"
            arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(img_dir / f"{stem}.png")
            m = (np.random.randint(0, n_classes, (size, size))).astype(np.uint8)
            Image.fromarray(m).save(mask_dir / f"{stem}.png")
        if with_meta:
            meta = {
                "name": ds,
                "classes": {str(c): f"class_{c}" for c in range(n_classes)},
                "background_class": 0,
            }
            with open(root / ds / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)


def test_loads_multi_dataset(tmp_path):
    _build_toy_root(tmp_path, ["dataset_A", "dataset_B"], n_per=4)
    ds = LabeledEMDataset(str(tmp_path))
    assert len(ds) == 8
    assert set(ds.dataset_meta.keys()) == {"dataset_A", "dataset_B"}


def test_getitem_shapes_and_types(tmp_path):
    _build_toy_root(tmp_path, ["dataset_A"], n_per=2, size=48)
    ds = LabeledEMDataset(str(tmp_path))
    image, mask, meta = ds[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.int64
    assert mask.shape == (48, 48)
    assert meta["dataset_name"] == "dataset_A"
    assert meta["background_class"] == 0
    assert "stem" in meta


def test_grayscale_image_converted_to_rgb(tmp_path):
    img_dir = tmp_path / "ds" / "images"
    mask_dir = tmp_path / "ds" / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.fromarray((np.random.rand(32, 32) * 255).astype(np.uint8), mode="L").save(
        img_dir / "a.png"
    )
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(mask_dir / "a.png")
    ds = LabeledEMDataset(str(tmp_path))
    image, mask, _ = ds[0]
    assert image.mode == "RGB"
    assert mask.shape == (32, 32)


def test_rgb_mask_takes_first_channel(tmp_path):
    img_dir = tmp_path / "ds" / "images"
    mask_dir = tmp_path / "ds" / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.fromarray((np.random.rand(16, 16, 3) * 255).astype(np.uint8)).save(
        img_dir / "a.png"
    )
    rgb_mask = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb_mask[..., 0] = 2  # class id in channel 0
    Image.fromarray(rgb_mask).save(mask_dir / "a.png")
    ds = LabeledEMDataset(str(tmp_path))
    _, mask, _ = ds[0]
    assert mask.dim() == 2
    assert int(mask.unique().item()) == 2


def test_only_common_stems_paired(tmp_path):
    img_dir = tmp_path / "ds" / "images"
    mask_dir = tmp_path / "ds" / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for stem in ["a", "b", "c"]:
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(img_dir / f"{stem}.png")
    for stem in ["a", "b"]:  # no mask for c
        Image.fromarray(np.zeros((16, 16), np.uint8)).save(mask_dir / f"{stem}.png")
    ds = LabeledEMDataset(str(tmp_path))
    assert len(ds) == 2


def test_works_without_meta(tmp_path):
    _build_toy_root(tmp_path, ["ds"], n_per=2, with_meta=False)
    ds = LabeledEMDataset(str(tmp_path))
    assert len(ds) == 2
    assert ds.dataset_meta["ds"]["background_class"] == 0


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LabeledEMDataset(str(tmp_path / "nope"))


def test_empty_root_raises(tmp_path):
    (tmp_path / "empty_ds").mkdir()
    with pytest.raises(RuntimeError):
        LabeledEMDataset(str(tmp_path))


def test_dataset_filter(tmp_path):
    _build_toy_root(tmp_path, ["dataset_A", "dataset_B"], n_per=3)
    ds = LabeledEMDataset(str(tmp_path), datasets=["dataset_A"])
    assert len(ds) == 3
    assert set(ds.dataset_meta.keys()) == {"dataset_A"}
