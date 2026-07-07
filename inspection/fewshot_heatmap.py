# Copyright (c) 2026.
#
# Class-evidence heatmap (CAM류) — frozen DINO feature + 선형 분류기 기준.
#
# 분류기가 CLS + patch-mean concat feature 위에서 도므로, 클래스 점수를
# 패치별로 정확히 분해할 수 있다:
#
#   score_c = W_cls·cls + mean_p( W_patch·patch_p ) + b
#                          └────────────┬──────────┘
#               각 패치 p 의 기여도 = W_patch·patch_p
#
# → 패치별 기여도를 h×w 격자로 reshape → 업샘플 → 원본에 overlay.
#   logreg / NCM 모두 동일하게 분해된다 (NCM 은 W = 클래스 prototype).
#
# ※ feature 전체 L2 정규화(1/||v||)는 이미지마다 양의 상수라 공간 패턴/argmax 에
#   영향 없음 → heatmap 에서는 무시한다 (상대 분포만 본다).
#
# 실행 (서버):
#   python inspection/fewshot_heatmap.py \
#       --config-file dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml \
#       --pretrained-weights /path/to/teacher_checkpoint.pth \
#       --data-root /path/to/class_folders \
#       --per-class 4 \
#       --output-dir ./out/heatmap

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]  # repo 루트 (inspection/ 상위)
for _p in (str(_REPO), str(_REPO / "dino_v3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dinov3.distributed as distributed  # noqa: E402
from dinov3.eval.setup import setup_and_build_model  # noqa: E402
from inspection.em_aug import build_em_eval_transform  # noqa: E402
from inspection.fewshot_separability import extract_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heatmap")


@torch.inference_mode()
def patch_grid_and_cls(model, image_tensor, n_blocks, autocast_dtype):
    """단일 이미지 → (patch_grid[C,h,w], cls[C*n_blocks]). norm 적용된 토큰."""
    device = torch.cuda.current_device()
    x = image_tensor.unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=autocast_dtype):
        outs = model.get_intermediate_layers(
            x, n=n_blocks, reshape=True, return_class_token=True, norm=True
        )
    # outs: tuple of (patch[B,C,h,w], cls[B,C]) per block (last n)
    patch_grid = outs[-1][0][0].float().cpu()  # [C, h, w] (마지막 블록)
    cls = torch.cat([ct[0] for (_, ct) in outs], dim=-1).float().cpu()  # [C*n]
    return patch_grid, cls


def build_classifier_weights(model, data_root, image_size, n_blocks, feature_kind,
                             autocast_dtype, clf_kind, batch_size, num_workers):
    """data_root 전체로 분류기를 학습/구성하고, 패치-분기 가중치 W_patch[C, embed_dim] 반환."""
    eval_tf = build_em_eval_transform(image_size)
    dataset = datasets.ImageFolder(data_root, transform=eval_tf)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    x, y = extract_features(model, loader, n_blocks, feature_kind, autocast_dtype)
    num_classes = len(dataset.classes)
    embed_dim = model.embed_dim
    # concat feature 의 마지막 embed_dim 열이 patch-mean 분기.
    assert feature_kind == "concat", "heatmap 은 --feature concat 이어야 patch 분기를 분해할 수 있음"

    if clf_kind == "ncm":
        # 클래스별 평균(=prototype)을 분류기 가중치로 사용
        W = np.zeros((num_classes, x.shape[1]), dtype=np.float32)
        for c in range(num_classes):
            m = x[y == c].mean(axis=0)
            W[c] = m / (np.linalg.norm(m) + 1e-8)
        bias = np.zeros(num_classes, dtype=np.float32)
    else:  # logreg
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(x, y)
        coef = clf.coef_  # [1, D] (2-class) or [C, D]
        if coef.shape[0] == 1:  # 이진 → 클래스 0 = -coef, 클래스 1 = +coef
            W = np.vstack([-coef[0], coef[0]]).astype(np.float32)
            bias = np.array([-clf.intercept_[0], clf.intercept_[0]], dtype=np.float32)
        else:
            W = coef.astype(np.float32)
            bias = clf.intercept_.astype(np.float32)

    w_patch = W[:, -embed_dim:]  # [C, embed_dim]
    # 전체 feature(concat) 로 예측까지 하려면 W 전체도 필요
    return dataset, W, bias, w_patch, num_classes


def overlay_heatmap(orig_pil, heat, out_path, title, alpha=0.5):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        logger.warning(f"matplotlib 없음, heatmap 스킵: {e}")
        return
    img = np.asarray(orig_pil.convert("L"), dtype=np.float32) / 255.0
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("input")
    ax[0].axis("off")
    ax[1].imshow(img, cmap="gray")
    vmax = float(np.percentile(np.abs(heat), 99)) + 1e-8  # robust (아웃라이어 무시)
    hm = ax[1].imshow(heat, cmap="seismic", vmin=-vmax, vmax=vmax, alpha=alpha)
    ax[1].set_title(title)
    ax[1].axis("off")
    fig.colorbar(hm, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="class-evidence heatmap (frozen DINO + 선형 분류기)")
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--pretrained-weights", required=True)
    ap.add_argument("--data-root", required=True, help="분류기 학습용 ImageFolder")
    ap.add_argument("--query-dir", default=None, help="heatmap 뽑을 이미지 폴더(미지정 시 data-root 에서 샘플)")
    ap.add_argument("--per-class", type=int, default=4, help="query 미지정 시 클래스당 시각화 장수")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--n-blocks", type=int, default=1)
    ap.add_argument("--clf", choices=["logreg", "ncm"], default="logreg")
    ap.add_argument("--target", choices=["pred", "true"], default="pred",
                    help="heatmap 의 대상 클래스: 예측(pred) 또는 정답(true)")
    ap.add_argument("--mode", choices=["contrastive", "raw"], default="contrastive",
                    help="contrastive: W_patch[target]-mean(others) (판별 영역 부각). raw: DC만 제거")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--output-dir", default="./out/heatmap")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not distributed.is_enabled():
        distributed.enable(overwrite=True)

    logger.info("모델 로드 중...")
    model, ctx = setup_and_build_model(
        config_file=args.config_file, pretrained_weights=args.pretrained_weights, output_dir=args.output_dir
    )
    model.cuda().eval()
    autocast_dtype = ctx["autocast_dtype"]

    logger.info("분류기 구성 중...")
    dataset, W, bias, w_patch, num_classes = build_classifier_weights(
        model, args.data_root, args.image_size, args.n_blocks, "concat",
        autocast_dtype, args.clf, args.batch_size, args.num_workers
    )
    class_names = dataset.classes
    embed_dim = model.embed_dim
    eval_tf = build_em_eval_transform(args.image_size)

    # query 목록: (path, true_label)
    queries = []
    if args.query_dir:
        for p in sorted(Path(args.query_dir).rglob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                queries.append((str(p), -1))
    else:
        per_c = {c: 0 for c in range(num_classes)}
        for path, lab in dataset.samples:
            if per_c[lab] < args.per_class:
                queries.append((path, lab))
                per_c[lab] += 1
    logger.info(f"{len(queries)} 장 heatmap 생성...")

    for i, (path, true_lab) in enumerate(queries):
        pil = Image.open(path).convert("RGB")
        tensor = eval_tf(pil)
        patch_grid, cls = patch_grid_and_cls(model, tensor, args.n_blocks, autocast_dtype)  # [C,h,w], [C*n]
        C, h, w = patch_grid.shape

        # 전체 concat feature 로 예측
        patch_mean = patch_grid.reshape(C, -1).mean(dim=1)  # [C]
        feat = torch.cat([cls, patch_mean]).numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        scores = W @ feat + bias
        pred = int(scores.argmax())

        target = pred if args.target == "pred" or true_lab < 0 else true_lab

        # 패치별 기여도. contrastive: 다른 클래스 대비 판별 신호 (공통 DC 제거).
        if args.mode == "contrastive" and num_classes > 1:
            others = np.delete(np.arange(num_classes), target)
            w_vec = w_patch[target] - w_patch[others].mean(axis=0)
        else:  # raw
            w_vec = w_patch[target]
        wp = torch.from_numpy(w_vec)  # [C]
        heat_grid = torch.einsum("chw,c->hw", patch_grid, wp).numpy()  # [h,w]
        if args.mode == "raw":
            heat_grid = heat_grid - heat_grid.mean()
        # 업샘플
        heat = torch.nn.functional.interpolate(
            torch.from_numpy(heat_grid)[None, None], size=(args.image_size, args.image_size),
            mode="bilinear", align_corners=False
        )[0, 0].numpy()

        disp = pil.resize((args.image_size, args.image_size))
        t_str = class_names[true_lab] if true_lab >= 0 else "?"
        title = f"target={class_names[target]} | pred={class_names[pred]} | true={t_str}"
        out_path = os.path.join(
            args.output_dir, f"{i:03d}_{Path(path).stem}_t-{class_names[target]}.png"
        )
        overlay_heatmap(disp, heat, out_path, title, alpha=args.alpha)

    logger.info(f"완료. 결과: {args.output_dir}")
    print(f"\nheatmap {len(queries)}장 저장 → {args.output_dir}")
    print("빨강=해당 클래스 지지 영역, 파랑=반대 (seismic, 0 중심)")


if __name__ == "__main__":
    main()
