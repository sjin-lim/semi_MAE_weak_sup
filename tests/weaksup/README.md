# Weak Supervision Unit Tests

HANDOFF_CONTEXT.md §12 의 "다음 작업" 1~4번(toy unit test, smoke test)을 커버.

## 실행

```bash
# torch + torchvision 있는 환경 (학습 서버 권장)
cd semi_MAE_weak_sup
pip install pytest          # 없으면
pytest tests/weaksup -v
```

torch 가 없는 환경에서는 각 테스트가 `importorskip` 으로 자동 skip 됨 (로컬 dev 머신).

## 구성

| 파일 | 대상 | 핵심 검증 |
|---|---|---|
| `test_losses.py` | `train/weaksup/losses.py` | random→~0, similar→0.3+, bounded[0,1], bg 제외, min_patches gating, gradient 흐름, self-curriculum 단조성, T sharpness, batch 집계 |
| `test_patch_label.py` | `data/labeled/patch_label.py` | grid 크기/개수, majority, purity threshold, remainder 처리, batched 일관성 |
| `test_dataset.py` | `data/labeled/dataset.py` | multi-dataset 로드, getitem shape/type, grayscale→RGB, RGB mask 첫 채널, stem 매칭, meta 유무, 에러 케이스 |
| `test_joint_transforms.py` | `data/labeled/joint_transforms.py` | image-mask geometric 동기, class id 보존(nearest), 2 global+N local 구조, mask→patch P 길이, view 독립성 |

## 주의

- `conftest.py` 가 `dino_v3/` 와 repo root 를 `sys.path` 에 추가 → `import dinov3.*`,
  `import util.*` 동작.
- `test_joint_transforms.py` 는 `FusedEMIntensity`(dinov3.data.augmentations) 의존 →
  해당 모듈의 무거운 deps(cv2/skimage 등) 없으면 skip.
- 서버에서 가장 먼저 확인할 것: `test_joint_transforms.py::test_joint_geometric_size_sync`
  (mask 가 v2 통과 후 2D 유지되는지 — WEAKSUP_REVIEW.md R3) 와
  `test_pipeline_mask_to_patch_labels_length` (patch 개수 P 일치 — R1).
