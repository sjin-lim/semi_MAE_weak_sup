# Per-patch pair-wise HINGE ANTI-MERGE loss for EM-domain weak supervision.
#
# Goal: backbone 이 visually-similar-but-semantically-different class pair
#       (예: 옅은 경계로 갈리는 phase) 를 *합치지 못하게* 막는다. 분리를 능동적으로
#       강요하지 않고, "라벨상 다른데 feature 가 합쳐진(sim>margin) 쌍" 만 margin 까지
#       떼어놓는다 → iBOT 가 묶은 구조 보존 + over-separation/scatter 회피.
#
# Design (검토 완료, 이전 exp-push 에서 전환):
#   - Per-patch (centroid 게임 방지)
#   - Pair-wise (push-only, SSL prototype 구조 보존)
#   - Class 0 제외 (background 정의가 dataset 마다 다름)
#   - HINGE: sim <= margin 인 쌍은 force 정확히 0 (이미 구분된 쌍 방치).
#            이전 normalized-exp 는 sim=0.5 에도 0.4 force 가 남아 전역 교란 → 폐기.
#   - Bounded [0, 1]: (relu(sim-margin)/(1-margin))^power 라 sim=1 → 1.0, sim<=margin → 0.

import logging
from typing import List

import torch
import torch.nn.functional as F

logger = logging.getLogger("dinov3")


def per_patch_pairwise_loss(
    features: torch.Tensor,        # (N_patches, D) student patch features
    labels: torch.Tensor,          # (N_patches,) class labels (-1 = ignore)
    margin: float = 0.85,
    power: float = 2.0,
    background_class: int = 0,
    min_patches_per_class: int = 4,
    skip_background: bool = True,
) -> torch.Tensor:
    """Hinge anti-merge per-patch loss.

    For each non-background class pair in the image:
        - Compute all cross-class patch sim matrix S_ij (cosine).
        - Penalize ONLY pairs above margin:  ℓ_ij = (relu(S_ij - margin) / (1-margin))^power
        - sim <= margin → 0 (이미 충분히 구분된 쌍은 건드리지 않음).
        - Self-curriculum: 분리되며 margin 아래로 내려가면 자동으로 force 소멸.

    Args:
        features: (N, D) student patch features, post-backbone.
        labels:   (N,) patch-level class labels. -1 means ignore.
        margin:   이 cosine 위로 붙은 cross-class 쌍만 떼어냄. 아래는 force=0.
                  0.85 → "0.85 이하면 이미 구분됨" 으로 간주.
        power:    hinge 의 차수. 2.0(제곱) → margin 근처에서 force 가 0 에서 부드럽게 시작
                  (1.0 선형은 margin 통과 쌍을 급격히 때려 iBOT 교란 위험).
        background_class: class id to treat as background (excluded).
        min_patches_per_class: skip classes with fewer than this many patches.
        skip_background: if True, exclude background_class from pair generation.

    Returns:
        scalar loss in [0, 1] (sim=1 saturated pair → 1.0, all-below-margin → 0).
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
    inv_range = 1.0 / max(1.0 - margin, 1e-6)   # normalize so sim=1 → 1.0

    pair_losses = []
    for i, c_a in enumerate(classes):
        feat_a = class_to_feats[c_a]
        for c_b in classes[i + 1:]:
            feat_b = class_to_feats[c_b]

            # Cross-class per-patch similarity matrix
            sim_matrix = feat_a @ feat_b.T  # (N_a, N_b)

            # Hinge: only sim > margin contributes; normalized to [0,1] then ^power.
            over = (sim_matrix - margin).clamp(min=0.0) * inv_range
            pair_loss = (over ** power).mean()

            pair_losses.append(pair_loss)

    if not pair_losses:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    return torch.stack(pair_losses).mean()


def batch_weak_loss(
    features_per_image: List[torch.Tensor],
    labels_per_image: List[torch.Tensor],
    margin: float = 0.85,
    power: float = 2.0,
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
        Returns 0 if no labeled image contributes (모든 쌍이 margin 아래 → 분리 완료).
    """
    if not features_per_image:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros((), device=device)

    device = features_per_image[0].device
    losses = []
    for feats, labels in zip(features_per_image, labels_per_image):
        loss_i = per_patch_pairwise_loss(
            feats, labels,
            margin=margin,
            power=power,
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
    margin: float = 0.85,
    background_class: int = 0,
    min_patches_per_class: int = 4,
    skip_background: bool = True,
) -> dict:
    """Image 내 class pair sim 분포 통계. 학습 중 logging 용.

    merged_pair_ratio = sim>margin 인 (즉 hinge force 가 실리는) pair 비율.
    학습이 진행되면 이 값이 0 으로 수렴해야 함 (분리 완료 = self-curriculum off).
    """
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
        "merged_pair_ratio": (pair_sims > margin).float().mean().item(),
    }
