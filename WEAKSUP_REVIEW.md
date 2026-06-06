# Weak Supervision 통합 로직 리뷰 (2026-06-06)

> 실행 환경(로컬)에 torch 가 없어 **정적 코드 리뷰**로 통합 경로를 추적한 결과.
> 학습/sanity test 는 서버에서 수행. 본 문서는 서버 실행 전 점검 체크리스트.

추적 경로:
`train.py do_train` → `RatioMixedLoader` → labeled `DataLoader` (collate) →
`SSLMetaArch.forward_backward(is_labeled=True)` → `per_patch_pairwise_loss`.

---

## ✅ 정상 확인된 부분

1. **Patch feature 추출 경로**
   `student_global["patch_pre_head"]` (shape `[n_global, B, P, D]`, backbone post /
   head pre) 가 weak loss 입력으로 정확히 연결됨
   ([ssl_meta_arch.py:438](dino_v3/dinov3/train/ssl_meta_arch.py#L438), 정의
   [:646](dino_v3/dinov3/train/ssl_meta_arch.py#L646)). 설계대로 **backbone level** 적용.

2. **Ordering 일치 (가장 중요)**
   - collate: `patch_labels = cat([crop0의 B개, crop1의 B개])`
     ([integration.py:99-101](dino_v3/dinov3/train/weaksup/integration.py#L99))
   - feature: `stu_patches.reshape(n_g*B, P, D)` → `[n_g, B]` flatten = `[crop0×B, crop1×B]`
     ([ssl_meta_arch.py:440](dino_v3/dinov3/train/ssl_meta_arch.py#L440))
   - 둘 다 crop-major 순서로 일치. ✅

3. **단일 backward**
   `compute_losses` 는 loss 만 반환, `backprop_loss(loss_accumulator)` 가 마지막에
   한 번만 호출. weak loss 는 `loss_accumulator += lambda_w * L_weak` 로 합산 후 동일
   backward → gradient 가 backbone 까지 정상 전파. ✅

4. **FSDP / distributed 동기 안전**
   `RatioMixedLoader` 의 labeled/unlabeled 분기는 `rng = Random(cfg.train.seed)` 로
   결정 — **모든 rank 가 동일 seed → 매 step 동일 경로 선택**. 따라서 모든 rank 가
   같은 step 에 labeled forward 를 타서 collective(all-gather/all-reduce) deadlock 없음.
   ([mixed_loader.py:38](dino_v3/dinov3/train/weaksup/mixed_loader.py#L38)) ✅ (설계상 중요)

5. **Mask geometric 동기**
   `tv_tensors.Mask` 로 wrapping 하여 RandomResizedCrop/Flip 이 image 와 mask 에
   동시 적용, mask 는 nearest 로 class id 보존. intensity aug 는 image 만.
   ([joint_transforms.py:68-72](dino_v3/dinov3/data/labeled/joint_transforms.py#L68)) ✅

6. **Bounded loss & self-curriculum**
   normalized exp `(exp(T·sim)-1)/(exp(T)-1)` 는 sim∈[-1,1] 에서 [0,1] bounded,
   sim 낮아지면 자동 decay. ✅ (단위 테스트로 검증 — `tests/weaksup/test_losses.py`)

7. **예외 비은닉**
   weak_sup 분기 try/except 가 `logger.error` 후 `raise`
   ([ssl_meta_arch.py:498-500](dino_v3/dinov3/train/ssl_meta_arch.py#L498)) — 조용히
   삼키지 않음. ✅

---

## ⚠️ 서버 실행 전 반드시 확인 (잠재 crash / 무효화)

### R1. Patch 개수(P) 일치 — assertion 부재 [높음]
`ssl_meta_arch.py:444` 는 `patch_labels.shape[0] == stu_patches_flat.shape[0]`
(= 이미지 개수 `n_g*B`) 만 검사. **patch 차원 P 는 미검사.**
- mask→patch_labels 의 P = `(global_crops_size / student.patch_size)^2`
- feature 의 P = student 가 실제로 뱉는 patch token 수
- 두 값이 다르면 `per_patch_pairwise_loss` 안에서 `features[mask]` (boolean mask 길이
  불일치) 로 **런타임 crash**.
- **위험 조건**: `teacher_to_student_resolution_scale != 1.0`. 이 경우 mask 생성용
  `patch_size_for_mask` ([train.py:465](dino_v3/dinov3/train/train.py#L465)) 는 scale
  반영하지만, patch_labels 는 `cfg.student.patch_size` (scale 미반영,
  [integration.py:163](dino_v3/dinov3/train/weaksup/integration.py#L163)) 를 사용 →
  P 불일치 가능.
- Stage 1~3 config 는 scale 기본값(1.0)이라 현재는 안전. 하지만 방어적으로
  `assert patch_labels.shape[1] == P` 추가 권장.

### R2. `local_batch_size` 미전달 [중간]
`build_labeled_loader_from_cfg(...)` 호출 시 `local_batch_size` 인자 생략
([train.py:483-488](dino_v3/dinov3/train/train.py#L483)) → `collate_data_and_cast` 에
`local_batch_size=None` 전달. unlabeled 경로
(`build_multi_resolution_data_loader_from_cfg`)가 이 값을 어떻게 넘기는지 비교 필요.
- single-resolution(Stage 1·2)에서는 None 이 무해할 가능성이 높지만,
  **Stage 3 mixed-resolution([512,768])** 에서 의미가 달라질 수 있음.
- 확인: `collate_data_and_cast` 에서 `local_batch_size` 사용처가 labeled 단일 해상도에
  영향 없는지.

### R3. Mask shape 가 v2 통과 후 정말 2D 인가 [중간]
`mask_to_patch_labels` 는 `mask.dim()==2` 가 아니면 `ValueError`
([patch_label.py:26](dino_v3/dinov3/data/labeled/patch_label.py#L26)).
`tv_tensors.Mask((H,W))` → RandomResizedCrop → `.as_subclass(Tensor)` 가 (H,W) 를
유지한다는 가정. torchvision 버전에 따라 (1,H,W) 가 될 가능성 미세하게 존재.
- `tests/weaksup/test_joint_transforms.py::test_joint_geometric_size_sync` 가
  서버에서 이 가정을 검증함. **반드시 먼저 실행.**

### R4. Labeled global crop 해상도 = SSL 해상도 가정
labeled aug 는 `global_crops_size` 단일값만 사용
([integration.py:122-125](dino_v3/dinov3/train/weaksup/integration.py#L122),
list 면 `[0]`). Stage 3 가 global crops `[512,768]` mixed 라면 labeled 은 512 만 사용 →
unlabeled 와 해상도 분포 다름. 학습은 되지만 의도와 다를 수 있음. (HANDOFF §5 의
Stage 3 "mixed" 의도와 불일치 가능 — 확인 필요.)

---

## 🔧 튜닝 / 품질 권장 (crash 아님)

### Q1. `patch_purity_threshold: 0.0` → 0.8 권장
stage2 config 주석 자체가 0.8 권장
([stage2_ssl_weaksup.yaml:102](dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml#L102)).
0.0 이면 경계(boundary) patch 가 majority class 로 배정되어, 실제로 두 class 가 섞인
patch 가 cross-class pair 에 들어가 noise 가 됨. EM boundary 가 많으면 weak signal
오염. **0.8 로 시작 권장.**

### Q2. bf16 features → weak loss float32 캐스팅 권장
`compute_precision.param_dtype: bf16` 이면 `patch_pre_head` 가 bf16.
`exp(T·sim)` (T=8) 은 bf16 에서 정밀도 낮음. `per_patch_pairwise_loss` 입력 전
`stu_patches_flat.float()` 캐스팅 시 수치 안정. (gradient 는 다시 bf16 으로 흘러도 OK.)

### Q3. vertical_flips 불일치
`LabeledImageAugmentation` 기본 `vertical_flips=True`
([joint_transforms.py:98](dino_v3/dinov3/data/labeled/joint_transforms.py#L98)) 이고
integration 빌드 시 명시 전달 안 함 → labeled 은 vertical flip ON.
unlabeled SSL aug 의 vertical flip 설정과 일치하는지 확인. (EM 은 회전 불변이라 무해할
가능성 높지만 일관성 차원에서 점검.)

### Q4. Python per-image loop 의 `.item()` 동기
`forward_backward` 의 weak 분기가 이미지마다 `loss_i.item()`
([ssl_meta_arch.py:465](dino_v3/dinov3/train/ssl_meta_arch.py#L465)) 로 GPU sync.
labeled step 당 `n_g*B`(예: 32)회. 학습 속도 병목이면 `.item()` 제거하고 tensor 누적
후 한 번에 평가하도록 최적화 가능. (정확성 무관, 성능만.)

---

## 📋 서버 실행 순서 (권장)

1. `pip install pytest` 후 `pytest tests/weaksup -v`
   → losses / patch_label / dataset / joint_transforms 전부 통과 확인.
   특히 **R3** (mask 2D) 와 P 길이(`test_pipeline_mask_to_patch_labels_length`) 검증.
2. **Stage 1 sanity** (`weak_sup.enabled: false`, 200 step) → loss 곡선이 semi_MAE v7 과
   유사한지. (weak 경로 미진입 = 기존 DINOv3 와 동일해야 함)
3. **Stage 2 sanity** (toy labeled 10장, 200 step) → `weak_loss`, `lambda_w`,
   `pair_sim_max/mean` logging 확인. iBOT/DINO loss 발산 없는지.
4. 이상 시 HANDOFF §11 실패 모드 표 참조.

---

## 한 줄 결론

통합 코드는 **구조적으로 정상이며 단일 backward·rank 동기·ordering 모두 올바름.**
crash 위험은 **R1(P 길이 assertion 부재)** 와 **R3(mask 2D 가정)** 두 곳에 집중되며,
둘 다 `tests/weaksup` 가 서버에서 잡아줌. 품질은 **Q1(purity 0.8)** 이 가장 영향 큼.
