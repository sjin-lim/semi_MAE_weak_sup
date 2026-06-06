# Per-patch pair-wise normalized exponential loss for EM-domain weak supervision.
#
# Goal: backbone 이 visually-similar-but-semantically-different class pair
#       (예: EM 의 옅은 회색 차이로 구분되는 phase) 를 분리하도록 weak guide.
#
# Design choices (검토 완료):
#   - Per-patch (centroid 게임 방지)
#   - Pair-wise (push-only, SSL prototype 구조 보존)
#   - Class 0 제외 (background 정의가 dataset 마다 다름)
#   - Normalized exponential (bounded [0, 1], self-curriculum)
#   - Image 내 모든 pair (random sampling 보다 stable)

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


def per_patch_pairwise_loss(
    features: torch.Tensor,        # (N_patches, D) student patch features
    labels: torch.Tensor,          # (N_patches,) class labels (-1 = ignore)
    T: float = 8.0,
    background_class: int = 0,
    min_patches_per_class: int = 4,
    skip_background: bool = True,
) -> torch.Tensor:
    """Selective per-patch pair-wise hard negative loss.

    For each non-background class pair in the image:
        - Compute all cross-class patch sim matrix
        - Normalized exponential penalty: high sim → high loss, low sim → tiny loss
        - Self-curriculum: as sims drop, loss naturally decays

    Args:
        features: (N, D) student patch features, post-backbone.
        labels:   (N,) patch-level class labels. -1 means ignore.
        T:        sharpness of exponential. Half-max sim ≈ 1 - 0.693/T.
                  T=8 → half-max at sim 0.91 (위험 영역만 강한 force).
        background_class: class id to treat as background (excluded).
        min_patches_per_class: skip classes with fewer than this many patches.
        skip_background: if True, exclude background_class from pair generation.

    Returns:
        scalar loss in [0, 1] (bounded by normalized exponential).
    """
    valid = labels >= 0
    if skip_background:
        valid = valid & (labels != background_class)

    if valid.sum() < min_patches_per_class * 2:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    # Group patches by class
    class_to_feats = {}
    for c in labels[valid].unique().tolist():
        mask = labels == c
        if mask.sum() < min_patches_per_class:
            continue
        cf = features[mask]
        class_to_feats[c] = F.normalize(cf, dim=-1)  # (N_c, D), L2-normalized

    if len(class_to_feats) < 2:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    classes = sorted(class_to_feats.keys())
    max_val = math.exp(T) - 1.0

    pair_losses = []
    for i, c_a in enumerate(classes):
        feat_a = class_to_feats[c_a]
        for c_b in classes[i + 1:]:
            feat_b = class_to_feats[c_b]

            # Cross-class per-patch similarity matrix
            sim_matrix = feat_a @ feat_b.T  # (N_a, N_b)

            # Normalized exponential — bounded [0, 1]
            raw = torch.exp(T * sim_matrix) - 1.0
            normalized = (raw / max_val).clamp(min=0.0)
            pair_loss = normalized.mean()

            pair_losses.append(pair_loss)

    if not pair_losses:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    return torch.stack(pair_losses).mean()


def batch_weak_loss(
    features_per_image: List[torch.Tensor],
    labels_per_image: List[torch.Tensor],
    T: float = 8.0,
    background_class: int = 0,
    min_patches_per_class: int = 4,
    skip_background: bool = True,
) -> torch.Tensor:
    """Apply per_patch_pairwise_loss across labeled images in batch.

    Args:
        features_per_image: list of (N_patches, D) feature tensors, one per labeled image.
        labels_per_image:   list of (N_patches,) label tensors, one per labeled image.
        Other args: same as per_patch_pairwise_loss.

    Returns:
        scalar loss = mean over images that contribute non-zero loss.
        Returns 0 if no labeled image contributes.
    """
    if not features_per_image:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros((), device=device)

    device = features_per_image[0].device
    losses = []
    for feats, labels in zip(features_per_image, labels_per_image):
        loss_i = per_patch_pairwise_loss(
            feats, labels,
            T=T,
            background_class=background_class,
            min_patches_per_class=min_patches_per_class,
            skip_background=skip_background,
        )
        if loss_i.item() > 0:
            losses.append(loss_i)

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


# ============================================================
# Monitoring helpers (학습 중 health check 용)
# ============================================================

@torch.no_grad()
def compute_pair_sim_stats(
    features: torch.Tensor,
    labels: torch.Tensor,
    background_class: int = 0,
    min_patches_per_class: int = 4,
    skip_background: bool = True,
) -> dict:
    """Image 내 class pair sim 분포 통계. 학습 중 logging 용."""
    valid = labels >= 0
    if skip_background:
        valid = valid & (labels != background_class)

    if valid.sum() < min_patches_per_class * 2:
        return {}

    centroids = {}
    for c in labels[valid].unique().tolist():
        mask = labels == c
        if mask.sum() < min_patches_per_class:
            continue
        centroids[c] = F.normalize(features[mask].mean(0), dim=-1)

    if len(centroids) < 2:
        return {}

    cs = torch.stack(list(centroids.values()))
    sim_matrix = cs @ cs.T
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    pair_sims = sim_matrix[mask]

    return {
        "pair_sim_max":  pair_sims.max().item(),
        "pair_sim_min":  pair_sims.min().item(),
        "pair_sim_mean": pair_sims.mean().item(),
        "n_pairs":       pair_sims.numel(),
        "hard_pair_ratio": (pair_sims > 0.7).float().mean().item(),
    }
