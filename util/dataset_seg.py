# Segmentation dataset: pairs images and masks by matching filenames
# across separate image / mask folders.

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# Image extensions to scan
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def build_seg_pairs(image_dir, mask_dir):
    """Match image/mask files by stem (filename without extension).

    Returns list of (image_path, mask_path) tuples.
    """
    # Index mask files by stem
    mask_stems = {}
    for f in Path(mask_dir).iterdir():
        if f.suffix.lower() in _IMG_EXTS:
            mask_stems[f.stem] = str(f)

    pairs = []
    for f in sorted(Path(image_dir).iterdir()):
        if f.suffix.lower() in _IMG_EXTS:
            if f.stem in mask_stems:
                pairs.append((str(f), mask_stems[f.stem]))

    if not pairs:
        raise RuntimeError(
            f"No matching image/mask pairs found between "
            f"{image_dir} and {mask_dir}")

    return pairs


class SegmentationDataset(Dataset):
    """Segmentation dataset that loads image/mask pairs.

    Args:
        pairs: list of (image_path, mask_path) from build_seg_pairs().
        input_size: int, resize both image and mask to this square size.
        is_train: if True, apply random augmentation (flip, crop).
        instance_norm: if True, per-image normalization instead of ImageNet stats.
    """

    def __init__(self, pairs, input_size=448, is_train=True, instance_norm=False):
        self.pairs = pairs
        self.input_size = input_size
        self.is_train = is_train
        self.instance_norm = instance_norm

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _load_rgb(path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    @staticmethod
    def _load_mask(path):
        """Load mask as single-channel. Pixel values are class indices."""
        mask = Image.open(path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        return mask

    def _sync_transform(self, image, mask):
        """Apply identical random transforms to image and mask."""
        # Resize
        image = TF.resize(image, [self.input_size, self.input_size],
                          interpolation=TF.InterpolationMode.BICUBIC)
        mask = TF.resize(mask, [self.input_size, self.input_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random rotation (0, 90, 180, 270)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # To tensor
        image = TF.to_tensor(image)  # (3, H, W), float [0,1]
        mask = torch.from_numpy(np.array(mask, dtype='int64'))  # (H, W) long

        # Normalize image
        if self.instance_norm:
            mean = image.mean(dim=[1, 2], keepdim=True)
            std = image.std(dim=[1, 2], keepdim=True)
            image = (image - mean) / (std + 1e-6)
        else:
            image = TF.normalize(image,
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = self._load_rgb(img_path)
        mask = self._load_mask(mask_path)
        image, mask = self._sync_transform(image, mask)
        return image, mask
