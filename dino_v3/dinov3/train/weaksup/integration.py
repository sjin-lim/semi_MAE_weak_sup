# Integration helpers: labeled dataset → SSL-compatible loader.
#
# Goal: labeled image+mask → 기존 SSL collate 와 동일한 형식의 batch + patch labels.
#       이렇게 하면 ssl_meta_arch 의 forward_backward 가 기존 path 를 그대로 타고,
#       추가로 weak_sup loss 만 분기 처리.

import logging
from functools import partial
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from dinov3.data.collate import collate_data_and_cast

logger = logging.getLogger("dinov3")


class _LabeledSSLAdapter(Dataset):
    """Wrap LabeledEMDataset + LabeledImageAugmentation → SSL-style sample.

    Each __getitem__ returns:
        (
            dict (DINO sample format):
                "global_crops": [tensor, tensor]
                "local_crops":  [tensor, ...]
            target_tuple:
                ()  # empty (matching unlabeled target_transform)
            extra (for weak sup):
                "global_crop_masks": [tensor, tensor]  # mask per global crop
                "meta": {...}
        )

    The collate fn (`labeled_collate_with_weak_sup`) will then:
      - run standard collate_data_and_cast on (dict, target) → SSL batch dict
      - extract patch labels from masks → add to batch
    """

    def __init__(self, base_dataset, augmentation, patch_size, purity_threshold, ignore_label):
        self.base = base_dataset
        self.aug = augmentation
        self.patch_size = patch_size
        self.purity = purity_threshold
        self.ignore = ignore_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask, meta = self.base[idx]
        out = self.aug(image, mask)
        # SSL-format dict expected by collate_data_and_cast
        ssl_dict = {
            "global_crops": out["global_crops"],
            "local_crops": out["local_crops"],
        }
        target_tuple = ()
        extra = {
            "global_crop_masks": out["global_crop_masks"],
            "meta": meta,
        }
        return (ssl_dict, target_tuple, extra)


def labeled_collate_with_weak_sup(
    samples_list: List[tuple],
    base_collate_fn,
    patch_size: int,
    purity_threshold: float = 0.0,
    ignore_label: int = -1,
) -> Dict[str, Any]:
    """Collate labeled samples: standard SSL collate + patch_labels extraction.

    samples_list: list of (ssl_dict, target_tuple, extra_dict).
    """
    from dinov3.data.labeled.patch_label import mask_to_patch_labels

    # Convert to format expected by collate_data_and_cast: list of (ssl_dict, target)
    ssl_samples = [(s[0], s[1]) for s in samples_list]
    collated = base_collate_fn(ssl_samples)

    # Extract patch labels from each sample's global_crop_masks
    n_global = len(samples_list[0][0]["global_crops"])
    patch_labels_per_crop = [[] for _ in range(n_global)]

    for s in samples_list:
        extra = s[2]
        masks_per_crop = extra["global_crop_masks"]  # list of (H, W) tensors
        for i, m in enumerate(masks_per_crop):
            pl = mask_to_patch_labels(
                m if isinstance(m, torch.Tensor) else torch.as_tensor(m, dtype=torch.long),
                patch_size=patch_size,
                min_purity=purity_threshold,
                ignore_label=ignore_label,
            )
            patch_labels_per_crop[i].append(pl)

    # Stack: (B, n_patches) per global crop, then concat to (n_global * B, n_patches)
    stacked_per_crop = [torch.stack(plist, dim=0) for plist in patch_labels_per_crop]
    # Match collated_global_crops ordering: [crop_0 for all samples, crop_1 for all samples, ...]
    collated["patch_labels"] = torch.cat(stacked_per_crop, dim=0)  # (n_global * B, n_patches)
    collated["n_global_crops"] = n_global
    collated["is_labeled"] = True

    return collated


def build_labeled_loader_from_cfg(cfg, mask_generator, dtype, n_tokens, local_batch_size=None):
    """Build labeled DataLoader matching unlabeled SSL batch format.

    Returns: torch.utils.data.DataLoader (finite, shuffled) over labeled samples.
             Wrapped externally by RatioMixedLoader for infinite mixing.
    """
    from dinov3.data.labeled import LabeledEMDataset, LabeledImageAugmentation

    if not cfg.weak_sup.labeled_root:
        raise ValueError("cfg.weak_sup.labeled_root must be set when weak_sup.enabled=True")

    base_ds = LabeledEMDataset(
        cfg.weak_sup.labeled_root,
        min_nonbg_classes=int(getattr(cfg.weak_sup, "min_nonbg_classes", 0)),
    )

    aug = LabeledImageAugmentation(
        global_crops_size=(
            cfg.crops.global_crops_size if isinstance(cfg.crops.global_crops_size, int)
            else cfg.crops.global_crops_size[0]
        ),
        local_crops_size=(
            cfg.crops.local_crops_size if isinstance(cfg.crops.local_crops_size, int)
            else cfg.crops.local_crops_size[0]
        ),
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_scale=tuple(cfg.crops.global_crops_scale),
        local_crops_scale=tuple(cfg.crops.local_crops_scale),
        horizontal_flips=cfg.crops.horizontal_flips,
        clahe=getattr(cfg.crops, "clahe", False),
        intensity_aug_config=getattr(cfg.crops, "intensity_aug", None),
        instance_norm=getattr(cfg.crops, "instance_norm", False),
        mean=tuple(cfg.crops.rgb_mean),
        std=tuple(cfg.crops.rgb_std),
    )

    adapter = _LabeledSSLAdapter(
        base_ds, aug,
        patch_size=cfg.student.patch_size,
        purity_threshold=cfg.weak_sup.patch_purity_threshold,
        ignore_label=cfg.weak_sup.ignore_label,
    )

    # Standard SSL collate (same as unlabeled)
    base_collate = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype=dtype,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=local_batch_size,
    )

    full_collate = partial(
        labeled_collate_with_weak_sup,
        base_collate_fn=base_collate,
        patch_size=cfg.student.patch_size,
        purity_threshold=cfg.weak_sup.patch_purity_threshold,
        ignore_label=cfg.weak_sup.ignore_label,
    )

    loader = DataLoader(
        adapter,
        batch_size=cfg.weak_sup.batch_size_labeled,
        num_workers=cfg.weak_sup.num_workers_labeled,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=full_collate,
        persistent_workers=cfg.weak_sup.num_workers_labeled > 0,
    )

    logger.info(
        f"[weak_sup] Labeled loader: {len(base_ds)} samples, "
        f"batch={cfg.weak_sup.batch_size_labeled}, workers={cfg.weak_sup.num_workers_labeled}"
    )
    return loader
