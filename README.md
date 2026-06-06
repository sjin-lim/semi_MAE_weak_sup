# semi_MAE_weak_sup — DINOv3 EM with Weakly Supervised Hard Negative Mining

`semi_MAE/` 의 DINOv3 코드를 base 로, EM 도메인 특화 weakly supervised 확장 구현.

## 목적

EM 의 visually-similar-but-semantically-different class pair (옅은 회색 차이 등) 를
SSL 단독으로 분리 못 하는 한계를 weak supervision 으로 보완.

## 3-Stage Pipeline

```
Stage 1: Pure SSL (기존 DINO 그대로)
  └─ config: configs/train/stage1_ssl_only.yaml
  
Stage 2: SSL + Weak Supervised Per-Patch Pair-wise Loss
  └─ config: configs/train/weaksup/stage2_ssl_weaksup.yaml
  └─ Per-patch pair-wise normalized exponential
  └─ Class 0 (background) 제외
  └─ Resume from Stage 1 teacher

Stage 3: High-Res Adaptation
  └─ config: configs/train/weaksup/stage3_hires_adapt.yaml
  └─ Weak sup retained with smaller lambda
  └─ Resume from Stage 2 teacher
```

## 디렉토리 구조

```
dino_v3/dinov3/
├── data/
│   ├── labeled/                  # NEW — labeled dataset (image + mask)
│   │   ├── __init__.py
│   │   ├── dataset.py            # LabeledEMDataset
│   │   ├── joint_transforms.py   # image+mask 동기 augmentation
│   │   └── patch_label.py        # mask → patch labels
│   ├── (기존 unlabeled WebDataset 등)
├── train/
│   ├── weaksup/                  # NEW — weak supervised loss
│   │   ├── __init__.py
│   │   ├── losses.py             # Per-patch pair-wise loss
│   │   └── mixed_loader.py       # labeled + unlabeled mixing
│   ├── ssl_meta_arch.py          # 수정: weak_sup hook 추가
│   └── train.py                  # 수정: mixed batch 처리
└── configs/
    └── train/
        └── weaksup/              # NEW — stage configs
            ├── stage1_ssl_only.yaml
            ├── stage2_ssl_weaksup.yaml
            └── stage3_hires_adapt.yaml
```

## 사용법

```bash
# Stage 1: pure SSL (기존 DINO 학습과 동일 동작)
python dino_v3/dinov3/train/train.py \
    --config-file dino_v3/dinov3/configs/train/weaksup/stage1_ssl_only.yaml \
    train.dataset_path="WebDataset:path=..." \
    train.output_dir=./out/stage1

# Stage 2: + weak sup
python dino_v3/dinov3/train/train.py \
    --config-file dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml \
    train.dataset_path="WebDataset:path=..." \
    weak_sup.labeled_root=/path/to/labeled_data \
    student.resume_from_teacher_chkpt=./out/stage1/.../teacher_checkpoint.pth \
    train.output_dir=./out/stage2

# Stage 3: high-res
python dino_v3/dinov3/train/train.py \
    --config-file dino_v3/dinov3/configs/train/weaksup/stage3_hires_adapt.yaml \
    train.dataset_path="WebDataset:path=..." \
    weak_sup.labeled_root=/path/to/labeled_data \
    student.resume_from_teacher_chkpt=./out/stage2/.../teacher_checkpoint.pth \
    train.output_dir=./out/stage3
```

## Weak Sup Loss 요약

```python
L_total = L_dino + L_ibot + λ_K · L_koleo                          # 모든 batch
        + λ_W · L_weak                                              # labeled image만

L_weak = (1/|P|) Σ_{(c_a, c_b)} mean(  (exp(T·sim_patch) - 1) / (exp(T) - 1)  )

  - 각 image 의 non-bg class pair (class > 0) 만
  - Per-patch sim matrix (centroid 게임 방지)
  - Normalized exponential (bounded [0, 1])
  - Self-curriculum (sim 낮으면 loss 자동 감소)
```

Hyperparameters: T=8 (sharpness), λ_W=20~30 (weight).

## 의존성

`semi_MAE/util/` (PerImageNormalize 등) 코드는 본 repo 에 복사되어 있음.
Original repo (`../semi_MAE/`) 에 종속되지 않음.

## 차별점 (vs 원본 semi_MAE)

| | semi_MAE | semi_MAE_weak_sup |
|---|---|---|
| MAE 코드 | 포함 (legacy) | 제외 |
| Labeled dataset | 없음 | LabeledEMDataset |
| Joint aug | 없음 | image+mask 동기 |
| Weak sup loss | 없음 | Per-patch pair-wise |
| Mixed batching | 없음 | labeled/unlabeled ratio |
| Stage 분리 | 없음 | Stage 1/2/3 config |

## Status

- [x] 코드 구조 복사
- [ ] Labeled dataset 구현
- [ ] Joint augmentation 구현
- [ ] Patch label collate 구현
- [ ] Weak sup loss 구현
- [ ] Mixed batch loader 구현
- [ ] Stage configs 작성
- [ ] Integration test (Stage 1 sanity)
- [ ] Sanity check (Stage 2 toy training)
