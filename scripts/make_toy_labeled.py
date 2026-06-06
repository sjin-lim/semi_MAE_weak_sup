#!/usr/bin/env python
"""Generate a toy labeled EM dataset for weak-sup sanity testing.

LabeledEMDataset 가 기대하는 폴더 구조를 합성 데이터로 생성:

    <out>/
    ├── dataset_A/
    │   ├── images/img_000.png ...   (RGB grayscale-like EM 모사)
    │   ├── masks/img_000.png  ...   (pixel value = class id, 0=bg)
    │   └── meta.json
    └── dataset_B/ ...

핵심: class 1 과 class 2 를 **옅은 회색 차이**로만 구분되게 만들어
(visually-similar-but-semantically-different) weak loss 에 실제 신호가 생기도록 함.
배경(class 0)은 또 다른 회색. blob 은 부드러운 원형 영역.

의존성: numpy, Pillow (torch 불필요 — 로컬에서도 생성 가능).

사용:
    python scripts/make_toy_labeled.py --out ./toy_labeled --n-per 10
    # → Stage 2 sanity: LABELED_ROOT=./toy_labeled
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def _add_blob(label_map, cx, cy, r, class_id, rng):
    """label_map 에 (cx,cy) 중심 반지름 r 원형 blob 을 class_id 로 채움."""
    H, W = label_map.shape
    ys, xs = np.ogrid[:H, :W]
    # 약간 울퉁불퉁한 경계 (boundary patch 생성 → purity threshold 테스트용)
    wobble = 1.0 + 0.15 * rng.standard_normal()
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= (r * wobble) ** 2
    label_map[mask] = class_id


def _render_image(label_map, gray_levels, rng, noise=12.0):
    """class id 별 gray level + Gaussian noise → RGB uint8 image."""
    img = np.zeros(label_map.shape, dtype=np.float32)
    for cid, level in gray_levels.items():
        img[label_map == cid] = level
    img += rng.normal(0, noise, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)  # grayscale → RGB


def make_dataset(ds_dir, n_per, size, n_classes, gray_levels, seed):
    img_dir = os.path.join(ds_dir, "images")
    mask_dir = os.path.join(ds_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(n_per):
        label_map = np.zeros((size, size), dtype=np.uint8)  # 0 = background
        # 각 non-bg class 마다 1~3개 blob
        for cid in range(1, n_classes):
            for _ in range(rng.integers(1, 4)):
                cx = int(rng.integers(size // 6, size - size // 6))
                cy = int(rng.integers(size // 6, size - size // 6))
                r = int(rng.integers(size // 10, size // 4))
                _add_blob(label_map, cx, cy, r, cid, rng)

        img = _render_image(label_map, gray_levels, rng)
        stem = f"img_{i:03d}"
        Image.fromarray(img).save(os.path.join(img_dir, f"{stem}.png"))
        Image.fromarray(label_map).save(os.path.join(mask_dir, f"{stem}.png"))

    meta = {
        "name": os.path.basename(ds_dir),
        "classes": {str(c): ("background" if c == 0 else f"phase_{c}") for c in range(n_classes)},
        "background_class": 0,
        "notes": "synthetic toy data for weak-sup sanity; classes 1/2 are gray-similar",
    }
    with open(os.path.join(ds_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[toy] {ds_dir}: {n_per} pairs, {n_classes} classes, gray={gray_levels}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./toy_labeled", help="labeled_root output dir")
    ap.add_argument("--n-datasets", type=int, default=2)
    ap.add_argument("--n-per", type=int, default=10, help="images per dataset")
    ap.add_argument("--size", type=int, default=448, help="image side (multiple of patch_size)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # dataset 마다 class 수를 다르게 (LabeledEMDataset 의 dynamic class 처리 검증)
    # 또한 class 1/2 의 gray level 을 가깝게 두어 weak loss 신호 생성.
    dataset_specs = [
        # (suffix, n_classes, gray_levels)
        ("A", 3, {0: 40, 1: 150, 2: 165}),   # 1 vs 2: 15 단계 차이 (옅은 회색)
        ("B", 4, {0: 30, 1: 120, 2: 135, 3: 200}),
        ("C", 3, {0: 50, 1: 100, 2: 110}),   # 1 vs 2: 10 단계 (더 어려움)
    ]
    specs = dataset_specs[: args.n_datasets]

    os.makedirs(args.out, exist_ok=True)
    for k, (suffix, n_classes, gray) in enumerate(specs):
        make_dataset(
            os.path.join(args.out, f"dataset_{suffix}"),
            n_per=args.n_per,
            size=args.size,
            n_classes=n_classes,
            gray_levels=gray,
            seed=args.seed + k,
        )
    total = len(specs) * args.n_per
    print(f"\n[toy] done. {total} pairs in {len(specs)} datasets → {args.out}")
    print(f"[toy] Stage 2 sanity: set LABELED_ROOT={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
