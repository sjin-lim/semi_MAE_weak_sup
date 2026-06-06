# Sanity Tests — 서버 실행 가이드

> 로컬엔 torch 가 없으므로 **학습 서버(GPU)** 에서 실행. 본격 학습 전 짧게 돌려
> "기존 DINOv3 와 동일 동작(Stage 1)" / "weak loss 정상 주입(Stage 2)" 을 확인한다.

선행: `pip install pytest && pytest tests/weaksup -v` (전부 pass 확인) → WEAKSUP_REVIEW.md.

---

## Stage 0. Unit tests (가장 먼저)

```bash
cd semi_MAE_weak_sup
pip install pytest      # 없으면
pytest tests/weaksup -v
```

전부 pass 해야 함. 특히:
- `test_joint_transforms.py::test_joint_geometric_size_sync` → mask 가 v2 통과 후 2D (리뷰 R3)
- `test_joint_transforms.py::test_pipeline_mask_to_patch_labels_length` → patch 개수 P (리뷰 R1)

---

## Stage 1 sanity — SSL only (weak_sup OFF)

**목적**: `weak_sup.enabled=false` 일 때 학습 코드가 기존 DINOv3 와 **100% 동일** 동작하는지.
weak 경로 미진입 확인 + loss 곡선이 semi_MAE 의 v7_combined 와 유사한지.

**필요**: WebDataset shards. (labeled 데이터 불필요.) 선택적으로 v4_ibot_ft/v7 teacher.

### 200-step 짧은 실행

```bash
cd semi_MAE_weak_sup

SHARDS_PATH=/path/to/webdataset/shards \
OUTPUT_DIR=./out/stage1_sanity \
NPROC=4 \
bash scripts/train_stage1.sh \
    optim.epochs=1 \
    train.OFFICIAL_EPOCH_LENGTH=200 \
    train.saveckp_freq=999 \
    train.compile=false
```

- `optim.epochs=1 × OFFICIAL_EPOCH_LENGTH=200` → `max_iter=200`.
- `compile=false` → 짧은 run 에서 컴파일 오버헤드 회피 (sanity 한정; 본 학습은 true).
- prior teacher 에서 출발하려면 `PRIOR_TEACHER=/path/to/teacher_checkpoint.pth` 추가.

### ✅ 통과 기준 (콘솔 / `out/stage1_sanity/training_metrics.json`)

| 확인 | 기대 |
|---|---|
| 로그에 `[weak_sup]` 라인 | **없어야 함** (enabled=false → RatioMixedLoader 미생성) |
| metric 키 `weak_loss`, `lambda_w` | **없어야 함** (weak 분기 미진입) |
| `loss_dino`, `loss_ibot`, `koleo` | 정상 출력, 감소 추세 |
| NaN | 없음 (있으면 HANDOFF §11) |
| `*_grad_norm` | clip(3.0) 한계 안 |
| step 속도 | semi_MAE Stage 와 유사 |

> 핵심: weak_sup OFF 에서는 **기존 코드 경로와 차이가 없어야** 한다. weak_loss 키가
> 하나라도 보이면 분기 로직 버그.

---

## Stage 2 sanity — SSL + Weak Sup (toy labeled)

> (이번 세션에서는 toy 데이터 생성까지 준비. 실행 가이드 상세는 다음 단계에서.)

### toy labeled 데이터 생성 (torch 불필요, 로컬에서도 가능)

```bash
python scripts/make_toy_labeled.py --out ./toy_labeled --n-per 10 --size 448
```

- `dataset_A`(3 class), `dataset_B`(4 class): class 1/2 를 **옅은 회색 차이**로 만들어
  weak loss 에 실제 신호가 생기도록 합성.
- 구조: `toy_labeled/dataset_*/images|masks/*.png` + `meta.json` (HANDOFF §9 형식).
- `--size` 는 `global_crops_size` 의 배수 권장(기본 448).

### 짧은 실행 (요지)

```bash
SHARDS_PATH=/path/to/shards \
LABELED_ROOT=$(pwd)/toy_labeled \
OUTPUT_DIR=./out/stage2_sanity \
STAGE1_TEACHER=/path/to/stage1/teacher_checkpoint.pth \
NPROC=4 \
bash scripts/train_stage2.sh \
    optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=200 \
    weak_sup.warmup_steps=20 train.compile=false
```

### ✅ 통과 기준 (요지)

| metric | 기대 |
|---|---|
| 로그 `[weak_sup] Mixed loader active` | 있어야 함 |
| `weak_loss` | labeled step 에서 `>0` logging (전부 0 이면 §11) |
| `lambda_w` | warmup 동안 0→λ_max 증가 |
| `pair_sim_max`/`pair_sim_mean` | logging 됨 |
| `loss_dino`/`loss_ibot` | 발산 안 함 |

상세 모니터링/실패 대응: HANDOFF_CONTEXT.md §10, §11.

---

## 적용된 리뷰 fix (참고)

- **R1**: `ssl_meta_arch.py` 에 patch 개수(P) 일치 assertion 추가 (resolution scale 불일치 조기 검출).
- **Q1**: stage2/stage3 config `patch_purity_threshold` 0.0 → **0.8** (boundary patch ignore).
- **Q2**: weak loss feature 를 float32 캐스팅 (bf16 exp 정밀도 보완).
