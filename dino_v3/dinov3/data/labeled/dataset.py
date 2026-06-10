# LabeledEMDataset — image + mask folder dataset for weak supervision.

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("dinov3")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _count_nonbg_classes(mask_path, background_class: int) -> int:
    """mask 의 background 제외 고유 class 개수 (RGB 면 첫 채널)."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return int(sum(1 for v in np.unique(arr).tolist() if v != background_class))


class LabeledEMDataset(Dataset):
    """Multi-dataset labeled EM data with image + mask + per-dataset meta.

    Expected folder structure:
        labeled_root/
        ├── dataset_A/
        │   ├── images/{stem}.{ext}
        │   ├── masks/{stem}.{ext}      # pixel value = class id
        │   └── meta.json (optional)
        ├── dataset_B/
        │   └── ...

    meta.json schema (optional, all fields optional):
        {
            "name": "dataset_A",
            "classes": {"0": "background", "1": "phase_alpha", ...},
            "background_class": 0,
            "ignore_indices": [],
            "notes": "..."
        }

    Each item returns: (image_PIL, mask_tensor, meta_dict)
    where meta_dict contains:
        - "dataset_name": str
        - "background_class": int (default 0)
        - "stem": str
        - "n_classes": int (count of unique classes in this dataset)
    """

    def __init__(
        self,
        labeled_root: str,
        datasets: Optional[list] = None,   # filter to specific subdirs; None = all
        min_nonbg_classes: int = 0,        # >=N non-bg class 없는 image 제외 (0=끔). pair 만들려면 2
    ):
        self.root = Path(labeled_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"labeled_root not found: {self.root}")

        # 두 layout 모두 지원:
        #   (A) flat:          labeled_root/images + masks
        #   (B) multi-dataset: labeled_root/<ds>/images + masks
        if (self.root / "images").is_dir() and (self.root / "masks").is_dir():
            candidates = [self.root]                      # flat layout
        else:
            candidates = sorted([p for p in self.root.iterdir() if p.is_dir()])
            if datasets is not None:
                candidates = [p for p in candidates if p.name in set(datasets)]

        self.entries = []
        self.dataset_meta = {}
        for ds_dir in candidates:
            img_dir = ds_dir / "images"
            mask_dir = ds_dir / "masks"
            if not (img_dir.is_dir() and mask_dir.is_dir()):
                logger.warning(f"[LabeledEMDataset] skip {ds_dir.name} (no images/ or masks/)")
                continue

            # Load meta
            meta_path = ds_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = {}
            meta.setdefault("name", ds_dir.name)
            meta.setdefault("background_class", 0)
            self.dataset_meta[ds_dir.name] = meta

            # Pair up images and masks by stem
            img_stems = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
            mask_stems = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
            common = sorted(set(img_stems) & set(mask_stems))

            bg = meta.get("background_class", 0)

            # non-bg class-count 필터 (옵션). 캐시(.nonbg_count.json)로 재스캔 회피.
            counts, cache_dirty = {}, False
            cache_path = ds_dir / ".nonbg_count.json"
            if min_nonbg_classes > 0 and cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        counts = json.load(f)
                except Exception:
                    counts = {}

            kept = skipped = 0
            for stem in common:
                if min_nonbg_classes > 0:
                    if stem not in counts:
                        counts[stem] = _count_nonbg_classes(mask_stems[stem], bg)
                        cache_dirty = True
                    if counts[stem] < min_nonbg_classes:
                        skipped += 1
                        continue
                self.entries.append({
                    "image_path": img_stems[stem],
                    "mask_path":  mask_stems[stem],
                    "dataset_name": ds_dir.name,
                    "background_class": bg,
                    "stem": stem,
                })
                kept += 1

            if min_nonbg_classes > 0 and cache_dirty:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(counts, f)
                except Exception as e:
                    logger.warning(f"[LabeledEMDataset] {ds_dir.name}: cache write 실패 ({e})")

            if min_nonbg_classes > 0:
                logger.info(
                    f"[LabeledEMDataset] {ds_dir.name}: kept {kept}, skipped {skipped} "
                    f"(<{min_nonbg_classes} non-bg class, bg={bg})"
                )
            else:
                logger.info(f"[LabeledEMDataset] {ds_dir.name}: {len(common)} pairs (bg={bg})")

        if not self.entries:
            raise RuntimeError(f"No labeled image-mask pairs found in {self.root}")
        logger.info(f"[LabeledEMDataset] total: {len(self.entries)} pairs from "
                    f"{len(self.dataset_meta)} datasets")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        image = Image.open(entry["image_path"]).convert("RGB")

        mask = Image.open(entry["mask_path"])
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            # Common case: mask stored as RGB; take first channel
            mask_arr = mask_arr[..., 0]
        mask_tensor = torch.from_numpy(mask_arr.astype(np.int64))

        meta = {
            "dataset_name": entry["dataset_name"],
            "background_class": entry["background_class"],
            "stem": entry["stem"],
        }
        return image, mask_tensor, meta
