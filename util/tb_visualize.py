"""TensorBoard image visualization utilities for MAE pre-training.

Usage (in main_pretrain.py):
    from util.tb_visualize import build_vis_batch, log_mae_reconstructions

    vis_batch = build_vis_batch(data_loader_train, num_samples=4, device=device)
    ...
    if epoch % args.vis_freq == 0:
        log_mae_reconstructions(model, vis_batch, log_writer, epoch,
                                mask_ratio=args.mask_ratio)
"""

import torch
import torchvision


def build_vis_batch(dataset, num_samples: int, device: torch.device) -> torch.Tensor:
    """데이터셋에서 num_samples 장을 고정 추출하여 반환.

    학습용 DataLoader와 별개로 SequentialSampler를 사용하므로
    DistributedSampler의 set_epoch 순서에 영향을 주지 않는다.
    """
    vis_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=num_samples,
        num_workers=0,   # 추가 worker 없이 메인 프로세스에서 로드
        drop_last=False,
    )
    samples, _ = next(iter(vis_loader))
    return samples[:num_samples].to(device).detach()


def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    """이미지별 min-max 정규화 → [0, 1] 범위로 변환 (TensorBoard 표시용)."""
    n = x.shape[0]
    mn = x.view(n, -1).min(dim=1).values.view(n, 1, 1, 1)
    mx = x.view(n, -1).max(dim=1).values.view(n, 1, 1, 1)
    return (x - mn) / (mx - mn + 1e-6)


@torch.no_grad()
def log_mae_reconstructions(
    model,
    samples: torch.Tensor,
    log_writer,
    epoch: int,
    mask_ratio: float = 0.75,
    tag: str = 'pretrain/reconstructions',
) -> None:
    """MAE 재구성 결과를 TensorBoard에 그리드 이미지로 기록한다.

    각 행: [원본(Original) | 마스크 입력(Masked) | 재구성(Reconstruction)]

    Args:
        model:       학습 중인 MAE 모델 (DDP 래핑 여부 무관).
        samples:     고정 시각화용 배치 텐서 (N, 3, H, W), 이미 device 위에 있어야 함.
        log_writer:  SummaryWriter 인스턴스.
        epoch:       현재 epoch (x축).
        mask_ratio:  마스킹 비율 (학습과 동일하게 맞춤).
        tag:         TensorBoard scalar 태그.
    """
    # DDP 래핑 해제
    model_core = model.module if hasattr(model, 'module') else model
    was_training = model_core.training
    model_core.eval()

    # ── Forward (학습과 동일하게 AMP autocast 적용 → OOM 방지) ──────────────
    with torch.cuda.amp.autocast():
        _, pred, mask = model_core(samples, mask_ratio=mask_ratio)
    rec = model_core.unpatchify(pred.float())   # (N, 3, H, W), FP32로 후처리

    # ── 마스크를 픽셀 공간으로 확장 ─────────────────────────────────────────
    patch_size = model_core.patch_embed.patch_size[0]
    H = samples.shape[2]
    n_side = H // patch_size                    # 한 변의 패치 수

    # mask: (N, L),  1 = 마스킹된 패치
    mask_2d = mask.reshape(-1, n_side, n_side)  # (N, nP, nP)
    mask_px = (
        mask_2d.unsqueeze(1)                    # (N, 1, nP, nP)
        .repeat_interleave(patch_size, dim=2)
        .repeat_interleave(patch_size, dim=3)   # (N, 1, H, W)
        .expand_as(samples)                     # (N, 3, H, W)
    )

    masked_input = samples.clone()
    masked_input[mask_px.bool()] = 0.0          # 마스크 영역 → 0 (정규화 기준 검정)

    # ── 표시용 정규화 ────────────────────────────────────────────────────────
    # 학습용 정규화(ImageNet 혹은 PerImageNormalize)와 무관하게 [0,1]로 리스케일
    orig_vis   = _minmax_normalize(samples)
    masked_vis = _minmax_normalize(masked_input)
    rec_vis    = _minmax_normalize(rec)

    # ── 그리드 구성: 원본 / 마스크 / 재구성 3열 ─────────────────────────────
    rows = []
    for i in range(samples.shape[0]):
        rows += [orig_vis[i], masked_vis[i], rec_vis[i]]

    grid = torchvision.utils.make_grid(
        torch.stack(rows), nrow=3, padding=2, pad_value=0.5
    )
    log_writer.add_image(tag, grid, epoch)

    if was_training:
        model_core.train()
