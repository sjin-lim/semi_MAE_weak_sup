# Copyright (c) 2026.
#
# Standalone separability / few-shot 진단 스크립트.
#
# 목적:
#   학습된 DINOv3(teacher) backbone 의 frozen feature 가 "폴더로 구분한 클래스"를
#   얼마나 잘 분리하는지 빠르게 검사한다. 분산(distributed) 하니스 없이 단일 GPU 로
#   동작하며, ImageFolder 레이아웃만 있으면 된다.
#
#   data_root/
#     classA/  img001.png img002.png ...
#     classB/  ...
#     classC/  ...
#
# 검사 항목:
#   1) NCM (Nearest Class Mean, cosine)   — 학습 0, weak-sup feature 분리도 sanity
#   2) kNN (cosine)                        — 학습 0
#   3) Logistic Regression (L2-norm feat)  — 10~50 shot 본 성능 (sklearn)
#   + silhouette / intra·inter cosine      — 비지도 분리도
#   + 혼동행렬, 클래스별 정확도
#   + 2D 임베딩 PNG (PCA, 가능하면 t-SNE)
#
# ★ 중요: 학습이 instance_norm(=PerImageNormalize) 으로 진행되므로 eval 정규화도
#         반드시 동일하게 맞춘다. ImageNet mean/std 를 쓰면 feature 가 망가져
#         분리 검사가 부당하게 실패한다.
#
# 실행 (서버):
#   cd dino_v3
#   python dinov3/eval/fewshot_separability.py \
#       --config-file dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml \
#       --pretrained-weights /path/to/teacher_checkpoint.pth \
#       --data-root /path/to/class_folders \
#       --image-size 448 \
#       --output-dir ./output_separability
#
#   pretrained-weights 는 consolidated .pth 또는 DCP 체크포인트 디렉토리 모두 가능.
#   config-file 은 학습에 쓴 것과 동일하게 (arch/n_storage_tokens 일치 목적).

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

# dinov3 패키지 import 가능하도록 경로 보정 (repo/dino_v3 가 sys.path 에 있어야 함)
_THIS = Path(__file__).resolve()
_DINOV3_ROOT = _THIS.parents[2]  # .../dino_v3
if str(_DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINOV3_ROOT))

import dinov3.distributed as distributed  # noqa: E402
from dinov3.eval.setup import setup_and_build_model  # noqa: E402
from dinov3.eval.em_aug import build_em_eval_transform, build_em_train_transform  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("separability")


# ----------------------------------------------------------------------------- #
# Feature 추출
# ----------------------------------------------------------------------------- #
@torch.inference_mode()
def extract_features(model, loader, n_blocks, feature_kind, autocast_dtype):
    """feature_kind: 'cls' | 'patchmean' | 'concat'"""
    feats, labels = [], []
    device = torch.cuda.current_device()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            outs = model.get_intermediate_layers(
                images, n=n_blocks, reshape=False, return_class_token=True, norm=True
            )
        # outs: tuple of (patch_tokens[B,N,C], cls_token[B,C]) per block (last n)
        cls = torch.cat([ct for (_, ct) in outs], dim=-1).float()  # [B, C*n]
        patch_mean = outs[-1][0].float().mean(dim=1)  # 마지막 블록 patch 평균 [B, C]
        if feature_kind == "cls":
            f = cls
        elif feature_kind == "patchmean":
            f = patch_mean
        else:  # concat
            f = torch.cat([cls, patch_mean], dim=-1)
        f = torch.nn.functional.normalize(f, dim=1, p=2)  # L2 정규화
        feats.append(f.cpu())
        labels.append(targets.clone())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


# ----------------------------------------------------------------------------- #
# 분류기 (NCM / kNN / Logistic) + 비지도 분리도
# ----------------------------------------------------------------------------- #
def ncm_predict(train_x, train_y, test_x, num_classes):
    # 클래스 평균 prototype (L2 정규화 feature 이므로 평균 후 재정규화) → cosine 최근접
    protos = np.zeros((num_classes, train_x.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        m = train_x[train_y == c].mean(axis=0)
        protos[c] = m / (np.linalg.norm(m) + 1e-8)
    sims = test_x @ protos.T  # cosine (둘 다 정규화됨)
    return sims.argmax(axis=1)


def run_episode(train_x, train_y, test_x, test_y, num_classes, knn_k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    res = {}
    # NCM
    pred = ncm_predict(train_x, train_y, test_x, num_classes)
    res["ncm"] = float((pred == test_y).mean())
    # kNN (cosine)
    k = min(knn_k, len(train_x))
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_x, train_y)
    res["knn"] = float((knn.predict(test_x) == test_y).mean())
    # Logistic Regression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(train_x, train_y)
    pred_lr = clf.predict(test_x)
    res["logreg"] = float((pred_lr == test_y).mean())
    res["_logreg_pred"] = pred_lr
    res["_logreg_test_y"] = test_y
    return res


def kshot_split(x, y, num_classes, shots, rng):
    """클래스당 shots 장을 support(train), 나머지를 query(test)."""
    tr_idx, te_idx = [], []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        tr_idx.extend(idx[:shots])
        te_idx.extend(idx[shots:])
    return np.array(tr_idx), np.array(te_idx)


def unsupervised_separability(x, y, num_classes):
    """silhouette(cosine) + 평균 intra/inter 클래스 cosine."""
    from sklearn.metrics import silhouette_score

    out = {}
    try:
        out["silhouette_cosine"] = float(silhouette_score(x, y, metric="cosine"))
    except Exception as e:  # noqa: BLE001
        out["silhouette_cosine"] = None
        logger.warning(f"silhouette 계산 실패: {e}")
    # 클래스 prototype 간 평균 cosine (낮을수록 클래스가 멀리 떨어짐)
    protos = []
    for c in range(num_classes):
        m = x[y == c].mean(axis=0)
        protos.append(m / (np.linalg.norm(m) + 1e-8))
    protos = np.stack(protos)
    sim = protos @ protos.T
    off = sim[~np.eye(num_classes, dtype=bool)]
    out["inter_class_cosine_mean"] = float(off.mean())
    out["inter_class_cosine_max"] = float(off.max())
    return out


def save_embedding_plot(x, y, class_names, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception as e:  # noqa: BLE001
        logger.warning(f"plot 스킵 (matplotlib/sklearn 없음): {e}")
        return

    emb = PCA(n_components=2).fit_transform(x)
    method = "PCA"
    if len(x) <= 5000:
        try:
            from sklearn.manifold import TSNE

            emb = TSNE(n_components=2, init="pca", perplexity=min(30, max(5, len(x) // 4))).fit_transform(x)
            method = "t-SNE"
        except Exception as e:  # noqa: BLE001
            logger.warning(f"t-SNE 실패, PCA 사용: {e}")

    plt.figure(figsize=(8, 7))
    for c, name in enumerate(class_names):
        m = y == c
        plt.scatter(emb[m, 0], emb[m, 1], s=12, label=name, alpha=0.7)
    plt.legend(markerscale=2, fontsize=8)
    plt.title(f"DINO feature {method} (클래스 분리도)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    logger.info(f"임베딩 plot 저장: {path}")


def confusion(pred, true, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    return cm


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="DINO feature 폴더-클래스 분리도 진단")
    ap.add_argument("--config-file", required=True, help="학습에 쓴 config yaml (arch 일치용)")
    ap.add_argument("--pretrained-weights", required=True, help="teacher 체크포인트(.pth 또는 DCP 디렉토리)")
    ap.add_argument("--data-root", required=True, help="ImageFolder 루트 (클래스별 하위 폴더)")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--feature", choices=["cls", "patchmean", "concat"], default="concat")
    ap.add_argument("--n-blocks", type=int, default=1, help="사용할 마지막 블록 수")
    ap.add_argument("--shots", type=int, default=None, help="클래스당 support 장수(few-shot). 미지정 시 50/50 stratified")
    ap.add_argument("--n-tries", type=int, default=5, help="에피소드 반복 수 (평균±std)")
    ap.add_argument("--test-frac", type=float, default=0.5, help="shots 미지정 시 test 비율")
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--output-dir", default="./output_separability")
    ap.add_argument("--seed", type=int, default=0)
    # ── train-time augmentation (support feature 증강) ──────────────────
    ap.add_argument("--train-aug", action="store_true",
                    help="support(train) 이미지에 augmentation view feature 를 추가해 분류기 학습")
    ap.add_argument("--aug-views", type=int, default=4, help="이미지당 augmentation view 수")
    ap.add_argument("--crop-scale", type=float, default=0.8, help="RandomResizedCrop scale 하한 (0.8배 크롭)")
    ap.add_argument("--noise-std-max", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 분산 초기화 (단일 프로세스 MANUAL 폴백 — torchrun 불필요).
    # setup_config 의 apply_scaling_rules_to_cfg 가 distributed.is_enabled() 를 assert 하므로
    # 모델 로드 전에 반드시 켜야 한다.
    if not distributed.is_enabled():
        distributed.enable(overwrite=True)

    # 1) 모델 (teacher backbone) 로드
    logger.info("모델 로드 중...")
    model, ctx = setup_and_build_model(
        config_file=args.config_file,
        pretrained_weights=args.pretrained_weights,
        output_dir=args.output_dir,
    )
    model.cuda().eval()
    autocast_dtype = ctx["autocast_dtype"]

    # 2) 데이터셋 (ImageFolder) — clean eval transform
    eval_tf = build_em_eval_transform(args.image_size)
    dataset = datasets.ImageFolder(args.data_root, transform=eval_tf)
    class_names = dataset.classes
    num_classes = len(class_names)
    counts = np.bincount([y for _, y in dataset.samples], minlength=num_classes)
    logger.info(f"클래스 {num_classes}개: " + ", ".join(f"{n}({c})" for n, c in zip(class_names, counts)))
    if num_classes < 2:
        raise SystemExit("클래스가 2개 미만입니다. data_root 하위에 클래스 폴더를 2개 이상 두세요.")

    def make_loader(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # 3) clean feature 추출 (query/test 및 train base view)
    logger.info(f"clean feature 추출 중 (kind={args.feature}, n_blocks={args.n_blocks})...")
    x, y = extract_features(model, make_loader(dataset), args.n_blocks, args.feature, autocast_dtype)
    logger.info(f"feature shape={x.shape}")

    # 3b) train augmentation feature bank (support 증강용). query 는 clean 유지.
    aug_x = None
    if args.train_aug:
        train_tf = build_em_train_transform(
            args.image_size, crop_scale=args.crop_scale, noise_std_max=args.noise_std_max
        )
        aug_dataset = datasets.ImageFolder(args.data_root, transform=train_tf)
        aug_loader = make_loader(aug_dataset)
        views = []
        for v in range(args.aug_views):
            logger.info(f"augmentation feature 추출 중... view {v + 1}/{args.aug_views}")
            xv, yv = extract_features(model, aug_loader, args.n_blocks, args.feature, autocast_dtype)
            assert np.array_equal(yv, y), "augmentation view 순서가 clean 과 불일치"
            views.append(xv)
        aug_x = np.stack(views, axis=1)  # [N, views, D]
        logger.info(f"aug bank shape={aug_x.shape}")

    # 4) 비지도 분리도
    sep = unsupervised_separability(x, y, num_classes)
    logger.info(f"비지도 분리도: {sep}")

    # 5) 에피소드 평가
    rng = np.random.default_rng(args.seed)
    keys = ["ncm", "knn", "logreg"]
    scores = {k: [] for k in keys}
    last = None
    for t in range(args.n_tries):
        if args.shots is not None:
            tr, te = kshot_split(x, y, num_classes, args.shots, rng)
        else:
            from sklearn.model_selection import train_test_split

            tr, te = train_test_split(
                np.arange(len(y)), test_size=args.test_frac, stratify=y, random_state=args.seed + t
            )
        # train(support) feature: clean + augmentation view들. query(test)는 clean.
        if aug_x is not None:
            train_x = np.concatenate([x[tr]] + [aug_x[tr, v] for v in range(args.aug_views)], axis=0)
            train_y = np.tile(y[tr], args.aug_views + 1)
        else:
            train_x, train_y = x[tr], y[tr]
        r = run_episode(train_x, train_y, x[te], y[te], num_classes, args.knn_k)
        for k in keys:
            scores[k].append(r[k])
        last = r
        logger.info(f"[try {t}] " + "  ".join(f"{k}={r[k]*100:.1f}" for k in keys))

    summary = {
        k: {"mean": float(np.mean(scores[k]) * 100), "std": float(np.std(scores[k]) * 100)} for k in keys
    }

    # 6) 마지막 에피소드 기준 혼동행렬 / 클래스별 정확도 (logreg)
    cm = confusion(last["_logreg_pred"], last["_logreg_test_y"], num_classes)
    per_class = {
        class_names[c]: float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None for c in range(num_classes)
    }

    # 7) 임베딩 plot
    save_embedding_plot(x, y, class_names, os.path.join(args.output_dir, "embedding.png"))

    # 8) 리포트 저장 + 출력
    report = {
        "config_file": args.config_file,
        "pretrained_weights": args.pretrained_weights,
        "feature": args.feature,
        "n_blocks": args.n_blocks,
        "image_size": args.image_size,
        "shots": args.shots,
        "n_tries": args.n_tries,
        "train_aug": args.train_aug,
        "aug_views": args.aug_views if args.train_aug else 0,
        "crop_scale": args.crop_scale,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_counts": counts.tolist(),
        "feature_dim": int(x.shape[1]),
        "unsupervised_separability": sep,
        "classifier_accuracy": summary,
        "logreg_per_class_accuracy": per_class,
        "logreg_confusion_matrix": cm.tolist(),
    }
    out_json = os.path.join(args.output_dir, "separability_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    aug_tag = f", train-aug x{args.aug_views}" if args.train_aug else ""
    print(f"  분리도 검사 결과 ({num_classes} classes, feature={args.feature}{aug_tag})")
    print("=" * 60)
    print(f"  silhouette(cosine)      : {sep.get('silhouette_cosine')}")
    print(f"  inter-class cosine mean : {sep['inter_class_cosine_mean']:.3f} (낮을수록 분리 잘됨)")
    print("  ----- 분류 정확도 (mean±std over tries) -----")
    for k in keys:
        print(f"    {k:8s}: {summary[k]['mean']:.1f} ± {summary[k]['std']:.1f}")
    print("  ----- logreg 클래스별 정확도 -----")
    for n, a in per_class.items():
        print(f"    {n:20s}: {'-' if a is None else f'{a*100:.1f}'}")
    print(f"\n  상세 리포트: {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
