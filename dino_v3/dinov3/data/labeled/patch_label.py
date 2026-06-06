# Patch-level class label extraction from pixel masks.

import torch


def mask_to_patch_labels(
    mask: torch.Tensor,        # (H, W) integer class id
    patch_size: int = 16,
    min_purity: float = 0.0,   # if > 0: patches with purity < min_purity → label -1 (ignore)
    ignore_label: int = -1,
) -> torch.Tensor:
    """Pixel-level mask → patch-level majority class label.

    Args:
        mask: (H, W) integer class id per pixel.
        patch_size: ViT patch size (default 16).
        min_purity: minimum class purity within a patch to keep its label.
                    purity = (majority class pixels) / (total pixels in patch).
                    Default 0.0 → always assign majority. Use 0.7~0.9 to exclude
                    ambiguous boundary patches.
        ignore_label: label assigned to patches below min_purity threshold.

    Returns:
        (n_patches_h * n_patches_w,) flattened patch labels.
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be 2D (H, W), got shape {tuple(mask.shape)}")

    H, W = mask.shape
    nH, nW = H // patch_size, W // patch_size

    # Reshape into patches: (nH, ps, nW, ps) → (nH, nW, ps*ps)
    patches = (
        mask[: nH * patch_size, : nW * patch_size]
        .reshape(nH, patch_size, nW, patch_size)
        .permute(0, 2, 1, 3)
        .reshape(nH, nW, -1)
    )  # (nH, nW, ps*ps)

    # Vectorized majority class via mode (PyTorch 2.0+ supports along dim)
    # Fallback: loop if mode unavailable
    try:
        majority, _ = patches.mode(dim=-1)
    except AttributeError:
        majority = torch.zeros((nH, nW), dtype=mask.dtype, device=mask.device)
        for i in range(nH):
            for j in range(nW):
                uniq, counts = patches[i, j].unique(return_counts=True)
                majority[i, j] = uniq[counts.argmax()]

    if min_purity > 0.0:
        # Compute purity = count of majority class / total patch pixels
        per_patch_size = patch_size * patch_size
        # For each patch, count how many pixels equal the majority value
        same = (patches == majority.unsqueeze(-1)).sum(dim=-1).float()
        purity = same / per_patch_size
        majority = torch.where(
            purity >= min_purity,
            majority,
            torch.full_like(majority, ignore_label),
        )

    return majority.flatten()


def patch_labels_batched(
    masks: torch.Tensor,       # (B, H, W)
    patch_size: int = 16,
    min_purity: float = 0.0,
    ignore_label: int = -1,
) -> torch.Tensor:
    """Batched version. Returns (B, n_patches)."""
    if masks.dim() != 3:
        raise ValueError(f"masks must be 3D (B, H, W), got shape {tuple(masks.shape)}")

    B = masks.shape[0]
    results = []
    for b in range(B):
        results.append(mask_to_patch_labels(masks[b], patch_size, min_purity, ignore_label))
    return torch.stack(results, dim=0)
