# Copyright (c) 2026.
#
# Feature Service 클라이언트 예시 (torch-free).
#
# 서버 /features 로 embedding 을 받아, 분류/등록을 **클라이언트에서** 수행한다.
# torch/GPU 불필요 — numpy + PIL + (requests 또는 urllib) 만 있으면 됨.
#
# 두 모드:
#   classify : 로컬 head.npz 로 이미지 분류
#     python service/client_example.py classify --server http://SERVER:8000 \
#         --input x.png --head head.npz
#   train    : 클래스별 폴더 → 서버로 embedding 수집 → DefectRegistry 구성/저장
#     python service/client_example.py train --server http://SERVER:8000 \
#         --data-root /path/to/class_folders --out registry.npz --clf logreg

import argparse
import io
import json
import sys
import urllib.request
import uuid
from pathlib import Path

import numpy as np

# inspection.em_classifier 의 torch-free 부분(ClassifierHead/DefectRegistry)만 사용
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for p in (str(_REPO), str(_REPO / "dino_v3")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inspection.em_classifier import ClassifierHead, DefectRegistry  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- #
# 서버 호출 — requests 있으면 사용, 없으면 urllib multipart 직접 구성
# --------------------------------------------------------------------------- #
def _post_multipart(url, file_items):
    """file_items: [(filename, bytes)]. 반환: 파싱된 JSON dict."""
    try:
        import requests

        files = [("images", (name, data, "application/octet-stream")) for name, data in file_items]
        r = requests.post(url, files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except ImportError:
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        for name, data in file_items:
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="images"; filename="{name}"\r\n'.encode())
            body.write(b"Content-Type: application/octet-stream\r\n\r\n")
            body.write(data)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            url, data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())


def get_features(server, image_paths, feature_kind=None, batch=16):
    """이미지 경로들 → (feats[N,D] np.float32, names[N]). 서버에 배치로 요청."""
    url = server.rstrip("/") + "/features"
    if feature_kind:
        url += f"?feature_kind={feature_kind}"
    feats, names = [], []
    paths = list(image_paths)
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        items = [(Path(p).name, Path(p).read_bytes()) for p in chunk]
        resp = _post_multipart(url, items)
        for r in resp["results"]:
            feats.append(np.asarray(r["feature"], dtype=np.float32))
            names.append(r["filename"])
    return np.vstack(feats), names


def _list_images(path):
    p = Path(path)
    if p.is_dir():
        return [str(q) for q in sorted(p.rglob("*")) if q.suffix.lower() in IMG_EXTS]
    return [str(p)]


# --------------------------------------------------------------------------- #
def cmd_classify(args):
    head = ClassifierHead.load(args.head)
    paths = _list_images(args.input)
    feats, names = get_features(args.server, paths, feature_kind=head.meta.get("feature_kind"))
    proba = head.predict_proba(feats)
    k = min(args.topk, len(head.class_names))
    for name, p in zip(names, proba):
        order = np.argsort(p)[::-1]
        tops = ", ".join(f"{head.class_names[j]}:{p[j]:.3f}" for j in order[:k])
        print(f"{name}\t-> {head.class_names[int(order[0])]} ({p[order[0]]:.3f})\t[{tops}]")


def cmd_train(args):
    """ImageFolder(클래스별 하위폴더) → 서버 embedding → DefectRegistry 구성/저장."""
    root = Path(args.data_root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise SystemExit(f"클래스 폴더 없음: {root}")

    # 서버 feature 설정을 meta 에 기록 (분류 시 일관성)
    import urllib.request as _u

    with _u.urlopen(args.server.rstrip("/") + "/health", timeout=30) as resp:
        model_info = json.loads(resp.read().decode()).get("model", {})

    reg = DefectRegistry(meta={"server": args.server, **model_info})
    for d in class_dirs:
        paths = _list_images(d)
        if not paths:
            print(f"  (건너뜀, 이미지 없음) {d.name}")
            continue
        feats, _ = get_features(args.server, paths, feature_kind=model_info.get("feature_kind"))
        reg.enroll(d.name, feats)
        print(f"  {d.name}: {len(paths)}장")

    reg_path = reg.save(args.out)
    print(f"registry 저장: {reg_path}  classes={reg.counts()}")

    head = reg.build_head(kind=args.clf, beta=args.beta)
    head_path = head.save(args.head_out or (str(Path(args.out).with_suffix("")) + "_head.npz"))
    print(f"head 저장: {head_path}  (classify 에 --head 로 사용)")


def main():
    ap = argparse.ArgumentParser(description="Feature Service 클라이언트 (torch-free)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="head.npz 로 이미지/폴더 분류")
    c.add_argument("--server", required=True, help="예: http://localhost:8000")
    c.add_argument("--input", required=True, help="이미지 파일 또는 폴더")
    c.add_argument("--head", required=True, help="ClassifierHead .npz")
    c.add_argument("--topk", type=int, default=3)
    c.set_defaults(func=cmd_classify)

    t = sub.add_parser("train", help="클래스별 폴더 → registry/head 구성")
    t.add_argument("--server", required=True)
    t.add_argument("--data-root", required=True, help="ImageFolder (클래스별 하위폴더)")
    t.add_argument("--out", required=True, help="registry 저장 경로(.npz)")
    t.add_argument("--head-out", default=None, help="head 저장 경로(기본: <out>_head.npz)")
    t.add_argument("--clf", choices=["logreg", "ncm", "tip"], default="logreg")
    t.add_argument("--beta", type=float, default=5.5, help="tip: Tip-Adapter 커널 sharpness")
    t.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
