# Unit tests for em_classifier: ClassifierHead(순수 numpy) 저장/로드/추론,
# pool_tokens(torch), end-to-end(EMClassifier) 통합(체크포인트 env 지정 시).
#
# 헤드 테스트는 torch 불필요 → 로컬에서도 numpy 만 있으면 실행됨.
# 통합 테스트는 EM_TEST_CONFIG / EM_TEST_CKPT / EM_TEST_DATA 환경변수 + CUDA 필요.

import os

import pytest

np = pytest.importorskip("numpy")

from dinov3.eval.em_classifier import ClassifierHead, _softmax, pool_tokens  # noqa: E402


# --------------------------------------------------------------------------- #
# 합성 feature: 클래스별로 다른 방향의 단위벡터 클러스터 (L2 정규화)
# --------------------------------------------------------------------------- #
def _synthetic(num_classes=3, per_class=40, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(num_classes, dim, dtype=np.float32)  # 직교 중심
    X, y = [], []
    for c in range(num_classes):
        pts = centers[c] + 0.15 * rng.standard_normal((per_class, dim)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
        X.append(pts)
        y.append(np.full(per_class, c))
    return np.concatenate(X), np.concatenate(y), [f"class{c}" for c in range(num_classes)]


# --------------------------------------------------------------------------- #
# 헤드 학습/추론
# --------------------------------------------------------------------------- #
def test_softmax_sums_to_one():
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    p = _softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_ncm_fit_predict_separable():
    X, y, names = _synthetic()
    head = ClassifierHead.fit(X, y, names, kind="ncm")
    assert head.num_classes == 3
    assert head.feature_dim == X.shape[1]
    acc = (head.predict(X) == y).mean()
    assert acc > 0.95, f"분리 가능한 데이터인데 NCM acc={acc}"


def test_logreg_fit_predict_separable():
    pytest.importorskip("sklearn")
    X, y, names = _synthetic()
    head = ClassifierHead.fit(X, y, names, kind="logreg")
    assert head.W.shape == (3, X.shape[1])
    acc = (head.predict(X) == y).mean()
    assert acc > 0.95


def test_logreg_binary_expands_to_two_rows():
    pytest.importorskip("sklearn")
    X, y, names = _synthetic(num_classes=2)
    head = ClassifierHead.fit(X, y, names, kind="logreg")
    assert head.W.shape[0] == 2 and head.bias.shape[0] == 2
    # 두 행은 부호 반대 (이진 확장 규약)
    assert np.allclose(head.W[0], -head.W[1])


def test_predict_proba_valid():
    X, y, names = _synthetic()
    head = ClassifierHead.fit(X, y, names, kind="ncm")
    proba = head.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_single_sample_1d_input():
    X, y, names = _synthetic()
    head = ClassifierHead.fit(X, y, names, kind="ncm")
    pred = head.predict(X[0])  # 1D 입력도 동작 (atleast_2d)
    assert pred.shape == (1,)


# --------------------------------------------------------------------------- #
# 저장/로드 roundtrip
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tmp_path):
    X, y, names = _synthetic()
    head = ClassifierHead.fit(X, y, names, kind="ncm", meta={"feature_kind": "concat", "n_blocks": 1})
    path = head.save(str(tmp_path / "head"))
    assert path.endswith(".npz") and os.path.exists(path)

    loaded = ClassifierHead.load(path)
    assert loaded.class_names == head.class_names
    assert np.allclose(loaded.W, head.W)
    assert np.allclose(loaded.bias, head.bias)
    assert loaded.meta["feature_kind"] == "concat"
    assert loaded.meta["clf_kind"] == "ncm"
    # 예측 동일
    assert np.array_equal(loaded.predict(X), head.predict(X))


def test_save_appends_npz(tmp_path):
    X, y, names = _synthetic()
    head = ClassifierHead.fit(X, y, names, kind="ncm")
    path = head.save(str(tmp_path / "noext"))
    assert path.endswith(".npz")


def test_shape_assertions():
    with pytest.raises(AssertionError):
        ClassifierHead(np.zeros((2, 4)), np.zeros(3), ["a", "b"])  # bias/class 수 불일치


# --------------------------------------------------------------------------- #
# pool_tokens (torch)
# --------------------------------------------------------------------------- #
def test_pool_tokens_concat_matches_manual():
    torch = pytest.importorskip("torch")
    B, n_blocks, N, C = 2, 1, 7, 16
    cls = torch.randn(B, C * n_blocks)
    patches = torch.randn(B, N, C)
    f = pool_tokens(cls, patches, "concat")
    # 수동 계산: concat([cls, patch_mean]) 후 L2 정규화
    manual = torch.cat([cls, patches.mean(dim=1)], dim=-1)
    manual = torch.nn.functional.normalize(manual, dim=1, p=2)
    assert torch.allclose(f, manual, atol=1e-5)
    assert f.shape == (B, C * (n_blocks + 1))
    # L2 norm == 1
    assert torch.allclose(f.norm(dim=1), torch.ones(B), atol=1e-5)


def test_pool_tokens_kinds():
    torch = pytest.importorskip("torch")
    cls = torch.randn(3, 16)
    patches = torch.randn(3, 5, 16)
    assert pool_tokens(cls, patches, "cls").shape == (3, 16)
    assert pool_tokens(cls, patches, "patchmean").shape == (3, 16)
    assert pool_tokens(cls, patches, "concat").shape == (3, 32)
    with pytest.raises(ValueError):
        pool_tokens(cls, patches, "bogus")


# --------------------------------------------------------------------------- #
# end-to-end 통합 (실제 백본 필요 — env 지정 시에만)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not all(os.environ.get(k) for k in ("EM_TEST_CONFIG", "EM_TEST_CKPT", "EM_TEST_DATA")),
    reason="EM_TEST_CONFIG/EM_TEST_CKPT/EM_TEST_DATA 미설정",
)
def test_end_to_end_fit_save_load_predict(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 필요")
    from PIL import Image
    from torchvision import datasets

    from dinov3.eval.em_classifier import EMClassifier, EMFeatureExtractor, fit_from_imagefolder

    config = os.environ["EM_TEST_CONFIG"]
    ckpt = os.environ["EM_TEST_CKPT"]
    data = os.environ["EM_TEST_DATA"]

    extractor = EMFeatureExtractor(config, ckpt, image_size=224, cache_dir=str(tmp_path / "cache"))
    head = fit_from_imagefolder(extractor, data, kind="logreg", num_workers=0)
    art = head.save(str(tmp_path / "art"))

    # 재로드 후 추론 (동일 extractor 재사용 위해 직접 결합)
    loaded = ClassifierHead.load(art)
    clf = EMClassifier(extractor, loaded)

    sample_path = next(p for p, _ in datasets.ImageFolder(data).samples)
    res = clf.predict(Image.open(sample_path).convert("RGB"), topk=2)
    assert res["label"] in head.class_names
    assert 0.0 <= res["score"] <= 1.0
    assert len(res["topk"]) == 2
