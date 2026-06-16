# Copyright (c) 2026.
#
# EM few-shot classification 모델 저장/로드/추론.
#
# 설계 (마이크로서비스 지향):
#   ┌─────────────────────────────────────────────────────────────┐
#   │ ClassifierHead   순수 numpy (W·x + b). torch 불필요.          │
#   │                  → 저장/로드/추론·단위테스트가 가볍다.        │
#   ├─────────────────────────────────────────────────────────────┤
#   │ EMFeatureExtractor  DINO 백본 wrapper (torch, lazy import).   │
#   │                     PIL → concat feature (학습과 동일 파이프).│
#   ├─────────────────────────────────────────────────────────────┤
#   │ EMClassifier        extractor + head 결합, end-to-end predict.│
#   └─────────────────────────────────────────────────────────────┘
#
# 아티팩트(.npz) 하나에 W/bias/클래스명 + feature 설정 + 백본 config/weights
# 경로까지 담는다 → `EMClassifier.from_artifact(path)` 한 번으로 추론 준비.
#
# torch 는 EMFeatureExtractor 안에서만 lazy import → 헤드만 쓰는 코드(서빙 헤드,
# 테스트)는 torch 없이도 동작.

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger("em_classifier")

# feature 종류 — fewshot_separability.py 와 동일 규약 유지할 것.
FEATURE_KINDS = ("cls", "patchmean", "concat")
ARTIFACT_VERSION = 1


# --------------------------------------------------------------------------- #
# 1) 헤드: 순수 numpy 분류기
# --------------------------------------------------------------------------- #
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ClassifierHead:
    """frozen feature 위의 선형 분류기. W:[C,D], bias:[C], class_names:[C].

    추론은 logits = X @ W.T + bias (순수 numpy). torch 불필요.
    """

    def __init__(self, W: np.ndarray, bias: np.ndarray, class_names: Sequence[str], meta: Optional[dict] = None):
        self.W = np.asarray(W, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.class_names = list(class_names)
        self.meta = dict(meta or {})
        assert self.W.ndim == 2, "W 는 [C, D]"
        assert self.W.shape[0] == len(self.class_names) == self.bias.shape[0]

    @property
    def num_classes(self) -> int:
        return self.W.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.W.shape[1]

    # ---- 학습 ----
    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, class_names: Sequence[str],
            kind: str = "logreg", meta: Optional[dict] = None) -> "ClassifierHead":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        num_classes = len(class_names)
        if kind == "ncm":
            # 클래스 평균 prototype (cosine 분류 → 보통 L2 정규화된 feature 가정)
            W = np.zeros((num_classes, X.shape[1]), dtype=np.float32)
            for c in range(num_classes):
                m = X[y == c].mean(axis=0)
                W[c] = m / (np.linalg.norm(m) + 1e-8)
            bias = np.zeros(num_classes, dtype=np.float32)
        elif kind == "logreg":
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(X, y)
            coef, intercept = clf.coef_, clf.intercept_
            if coef.shape[0] == 1:  # 이진 → [클래스0=-w, 클래스1=+w]
                W = np.vstack([-coef[0], coef[0]]).astype(np.float32)
                bias = np.array([-intercept[0], intercept[0]], dtype=np.float32)
            else:
                W = coef.astype(np.float32)
                bias = intercept.astype(np.float32)
        else:
            raise ValueError(f"알 수 없는 kind: {kind}")
        meta = dict(meta or {})
        meta["clf_kind"] = kind
        return cls(W, bias, class_names, meta)

    # ---- 추론 ----
    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return X @ self.W.T + self.bias  # [N, C]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.decision(X))

    # ---- 직렬화 ----
    def save(self, path: str) -> str:
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        meta = dict(self.meta)
        meta["_artifact_version"] = ARTIFACT_VERSION
        np.savez(
            path,
            W=self.W,
            bias=self.bias,
            class_names=np.array(self.class_names, dtype=object),
            meta=json.dumps(meta, ensure_ascii=False),
        )
        logger.info(f"헤드 저장: {path} (C={self.num_classes}, D={self.feature_dim})")
        return path

    @classmethod
    def load(cls, path: str) -> "ClassifierHead":
        npz = np.load(path, allow_pickle=True)
        meta = json.loads(str(npz["meta"]))
        return cls(npz["W"], npz["bias"], list(npz["class_names"]), meta)


# --------------------------------------------------------------------------- #
# 2) feature pooling (학습과 동일 규약) — torch 텐서 입력
# --------------------------------------------------------------------------- #
def pool_tokens(cls_tokens: "object", patch_tokens_last: "object", feature_kind: str):
    """get_intermediate_layers 출력에서 feature 벡터를 만든다 (L2 정규화 포함).

    cls_tokens: [B, C*n]  (마지막 n 블록 cls concat)
    patch_tokens_last: [B, N, C]  (마지막 블록 patch tokens)
    fewshot_separability.extract_features 와 동일한 규약.
    """
    import torch

    patch_mean = patch_tokens_last.float().mean(dim=1)  # [B, C]
    cls_tokens = cls_tokens.float()
    if feature_kind == "cls":
        f = cls_tokens
    elif feature_kind == "patchmean":
        f = patch_mean
    elif feature_kind == "concat":
        f = torch.cat([cls_tokens, patch_mean], dim=-1)
    else:
        raise ValueError(f"알 수 없는 feature_kind: {feature_kind}")
    return torch.nn.functional.normalize(f, dim=1, p=2)


# --------------------------------------------------------------------------- #
# 3) 백본 wrapper (torch, lazy import)
# --------------------------------------------------------------------------- #
class EMFeatureExtractor:
    """DINO teacher 백본 → concat feature. PIL/배치 입력."""

    def __init__(self, config_file: str, pretrained_weights: str, image_size: int = 448,
                 n_blocks: int = 1, feature_kind: str = "concat", cache_dir: str = "./.em_cache"):
        import torch  # noqa: F401

        import dinov3.distributed as distributed
        from dinov3.eval.setup import setup_and_build_model

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if not distributed.is_enabled():
            distributed.enable(overwrite=True)
        model, ctx = setup_and_build_model(
            config_file=config_file, pretrained_weights=pretrained_weights, output_dir=cache_dir
        )
        model.cuda().eval()
        self.model = model
        self.autocast_dtype = ctx["autocast_dtype"]
        self.image_size = image_size
        self.n_blocks = n_blocks
        self.feature_kind = feature_kind
        self.config_file = config_file
        self.pretrained_weights = pretrained_weights

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    def _transform(self):
        from dinov3.eval.em_aug import build_em_eval_transform

        return build_em_eval_transform(self.image_size)

    def features(self, images, batch_size: int = 32) -> np.ndarray:
        """images: PIL.Image 또는 그 리스트 → np.ndarray [B, D] (L2 정규화)."""
        import torch
        from PIL import Image

        if isinstance(images, Image.Image):
            images = [images]
        tf = self._transform()
        device = torch.cuda.current_device()
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                x = torch.stack([tf(im.convert("RGB")) for im in batch]).to(device)
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    outs = self.model.get_intermediate_layers(
                        x, n=self.n_blocks, reshape=False, return_class_token=True, norm=True
                    )
                cls = torch.cat([ct for (_, ct) in outs], dim=-1)
                f = pool_tokens(cls, outs[-1][0], self.feature_kind)
                out.append(f.float().cpu().numpy())
        return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# 4) end-to-end 분류기 (extractor + head)
# --------------------------------------------------------------------------- #
class EMClassifier:
    def __init__(self, extractor: EMFeatureExtractor, head: ClassifierHead):
        self.extractor = extractor
        self.head = head

    @classmethod
    def from_artifact(cls, artifact_path: str, config_file: Optional[str] = None,
                      pretrained_weights: Optional[str] = None, cache_dir: str = "./.em_cache") -> "EMClassifier":
        """헤드 아티팩트 로드 + 백본 구성 (경로는 아티팩트 meta 에서, 필요 시 override)."""
        head = ClassifierHead.load(artifact_path)
        m = head.meta
        extractor = EMFeatureExtractor(
            config_file=config_file or m["config_file"],
            pretrained_weights=pretrained_weights or m["pretrained_weights"],
            image_size=int(m.get("image_size", 448)),
            n_blocks=int(m.get("n_blocks", 1)),
            feature_kind=m.get("feature_kind", "concat"),
            cache_dir=cache_dir,
        )
        return cls(extractor, head)

    def predict(self, images, topk: int = 1, batch_size: int = 32) -> List[dict]:
        """images: PIL 또는 리스트 → [{label, score, topk:[(name,prob)...]}]"""
        from PIL import Image

        single = isinstance(images, Image.Image)
        if single:
            images = [images]
        feats = self.extractor.features(images, batch_size=batch_size)
        proba = self.head.predict_proba(feats)
        names = self.head.class_names
        results = []
        k = min(topk, len(names))
        for p in proba:
            order = np.argsort(p)[::-1]
            top = [(names[j], float(p[j])) for j in order[:k]]
            results.append({"label": names[int(order[0])], "score": float(p[order[0]]), "topk": top})
        return results[0] if single else results


# --------------------------------------------------------------------------- #
# 5) 학습 편의 함수 + CLI
# --------------------------------------------------------------------------- #
def fit_from_imagefolder(extractor: EMFeatureExtractor, data_root: str, kind: str = "logreg",
                         batch_size: int = 32, num_workers: int = 8) -> ClassifierHead:
    """ImageFolder 전체로 헤드 학습. 백본 feature 추출 후 numpy 헤드 fit."""
    import torch
    from torchvision import datasets

    from dinov3.eval.fewshot_separability import extract_features

    dataset = datasets.ImageFolder(data_root, transform=extractor._transform())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    X, y = extract_features(extractor.model, loader, extractor.n_blocks, extractor.feature_kind, extractor.autocast_dtype)
    meta = {
        "feature_kind": extractor.feature_kind,
        "n_blocks": extractor.n_blocks,
        "image_size": extractor.image_size,
        "embed_dim": extractor.embed_dim,
        "config_file": str(Path(extractor.config_file).resolve()),
        "pretrained_weights": str(Path(extractor.pretrained_weights).resolve()),
    }
    return ClassifierHead.fit(X, y, dataset.classes, kind=kind, meta=meta)


def _cli_fit(args):
    extractor = EMFeatureExtractor(
        args.config_file, args.pretrained_weights, image_size=args.image_size,
        n_blocks=args.n_blocks, feature_kind=args.feature, cache_dir=args.cache_dir,
    )
    head = fit_from_imagefolder(extractor, args.data_root, kind=args.clf,
                                batch_size=args.batch_size, num_workers=args.num_workers)
    out = head.save(args.out)
    print(f"저장됨: {out}  (classes={head.class_names})")


def _cli_predict(args):
    clf = EMClassifier.from_artifact(
        args.artifact, config_file=args.config_file, pretrained_weights=args.pretrained_weights,
        cache_dir=args.cache_dir,
    )
    from PIL import Image

    paths = []
    p = Path(args.input)
    if p.is_dir():
        for q in sorted(p.rglob("*")):
            if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                paths.append(q)
    else:
        paths.append(p)
    imgs = [Image.open(q).convert("RGB") for q in paths]
    res = clf.predict(imgs, topk=args.topk)
    if isinstance(res, dict):
        res = [res]
    for q, r in zip(paths, res):
        tops = ", ".join(f"{n}:{s:.3f}" for n, s in r["topk"])
        print(f"{q.name}\t-> {r['label']} ({r['score']:.3f})\t[{tops}]")


def main(argv=None):
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="EM few-shot classification 학습/추론")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="ImageFolder 로 헤드 학습 후 아티팩트 저장")
    f.add_argument("--config-file", required=True)
    f.add_argument("--pretrained-weights", required=True)
    f.add_argument("--data-root", required=True)
    f.add_argument("--out", required=True, help="저장 경로(.npz)")
    f.add_argument("--clf", choices=["logreg", "ncm"], default="logreg")
    f.add_argument("--feature", choices=list(FEATURE_KINDS), default="concat")
    f.add_argument("--n-blocks", type=int, default=1)
    f.add_argument("--image-size", type=int, default=448)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--num-workers", type=int, default=8)
    f.add_argument("--cache-dir", default="./.em_cache")
    f.set_defaults(func=_cli_fit)

    p = sub.add_parser("predict", help="아티팩트 로드 후 이미지/폴더 추론")
    p.add_argument("--artifact", required=True)
    p.add_argument("--input", required=True, help="이미지 파일 또는 폴더")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--config-file", default=None, help="백본 config override (기본: 아티팩트 meta)")
    p.add_argument("--pretrained-weights", default=None, help="백본 weights override")
    p.add_argument("--cache-dir", default="./.em_cache")
    p.set_defaults(func=_cli_predict)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
