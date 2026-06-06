# Weak Supervision Integration Notes

기존 DINOv3 학습 코드에 weak supervision 을 통합하기 위한 작업 가이드.

## 현재 상태 (Skeleton 완료)

### ✅ 작성 완료된 신규 모듈

```
dino_v3/dinov3/data/labeled/
├── __init__.py                  # exports
├── dataset.py                   # LabeledEMDataset (multi-dataset folder)
├── joint_transforms.py          # image+mask 동기 augmentation
└── patch_label.py               # mask → patch labels

dino_v3/dinov3/train/weaksup/
├── __init__.py                  # exports
├── losses.py                    # per_patch_pairwise_loss, batch_weak_loss
└── mixed_loader.py              # RatioMixedLoader

dino_v3/dinov3/configs/
├── ssl_default_config.yaml      # MODIFIED: weak_sup 섹션 추가
└── train/weaksup/
    ├── stage1_ssl_only.yaml     # weak_sup.enabled: false
    ├── stage2_ssl_weaksup.yaml  # weak_sup.enabled: true
    └── stage3_hires_adapt.yaml  # weak_sup.enabled: true, lambda 축소
```

### ⏳ 다음 단계 — train.py / ssl_meta_arch.py 통합

기존 학습 코드를 수정하여 weak_sup branch 를 추가해야 합니다.

## 필요한 코드 수정 위치

### 1. `dino_v3/dinov3/train/train.py`

#### `build_data_loader_from_cfg()` 부근

`weak_sup.enabled: True` 면 unlabeled loader 와 labeled loader 를 만들고
`RatioMixedLoader` 로 wrapping:

```python
# 기존
data_loader = make_data_loader(dataset=dataset, ...)

# 신규 (weak_sup.enabled: true 분기)
if cfg.weak_sup.enabled and cfg.weak_sup.labeled_root:
    from dinov3.data.labeled import LabeledEMDataset, LabeledImageAugmentation
    from dinov3.train.weaksup.mixed_loader import RatioMixedLoader
    
    labeled_ds = LabeledEMDataset(cfg.weak_sup.labeled_root)
    labeled_aug = LabeledImageAugmentation(
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        horizontal_flips=cfg.crops.horizontal_flips,
        clahe=cfg.crops.clahe,
        intensity_aug_config=getattr(cfg.crops, "intensity_aug", None),
        instance_norm=getattr(cfg.crops, "instance_norm", False),
    )
    
    # Wrap dataset with augmentation
    class _AugmentedLabeled(torch.utils.data.Dataset):
        def __init__(self, base, aug, patch_size, purity_threshold, ignore_label):
            self.base = base
            self.aug = aug
            self.patch_size = patch_size
            self.purity = purity_threshold
            self.ignore = ignore_label
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, mask, meta = self.base[i]
            out = self.aug(img, mask)  # global_crops + global_crop_masks + local_crops
            # mask → patch label
            from dinov3.data.labeled import mask_to_patch_labels
            out["patch_labels_g1"] = mask_to_patch_labels(
                out["global_crop_masks"][0], self.patch_size, self.purity, self.ignore
            )
            out["patch_labels_g2"] = mask_to_patch_labels(
                out["global_crop_masks"][1], self.patch_size, self.purity, self.ignore
            )
            out["meta"] = meta
            return out
    
    aug_labeled = _AugmentedLabeled(
        labeled_ds, labeled_aug,
        patch_size=cfg.student.patch_size,
        purity_threshold=cfg.weak_sup.patch_purity_threshold,
        ignore_label=cfg.weak_sup.ignore_label,
    )
    labeled_loader = torch.utils.data.DataLoader(
        aug_labeled,
        batch_size=cfg.weak_sup.batch_size_labeled,
        num_workers=cfg.weak_sup.num_workers_labeled,
        collate_fn=_labeled_collate,  # 별도 정의 필요
        shuffle=True, drop_last=True, pin_memory=True,
    )
    
    data_loader = RatioMixedLoader(
        unlabeled_loader=data_loader,
        labeled_loader=labeled_loader,
        labeled_ratio=cfg.weak_sup.labeled_ratio,
        seed=cfg.train.seed,
    )
```

#### Training loop 안 `forward_backward()` 호출 부근

`batch["is_labeled"]` flag 로 분기:

```python
for data in data_loader:
    is_labeled = data.get("is_labeled", False)
    
    # 기존 SSL forward_backward
    total_loss, metrics_dict = model.forward_backward(
        data, teacher_temp=teacher_temp, iteration=it,
        is_labeled=is_labeled,  # 신규 인자
    )
```

### 2. `dino_v3/dinov3/train/ssl_meta_arch.py`

#### `SSLMetaArch.forward_backward()` 수정

`is_labeled` 분기 추가:

```python
def forward_backward(self, data, *, teacher_temp, iteration=0, is_labeled=False, **kwargs):
    # ... 기존 SSL loss 계산 ...
    total_loss = sum(loss_dict.values())
    
    # Weak sup loss 추가 (labeled batch 만)
    if is_labeled and self.cfg.weak_sup.enabled:
        from dinov3.train.weaksup import per_patch_pairwise_loss
        
        # student backbone features 추출 (post-backbone, pre-head)
        # data["global_crops"] 의 student forward 결과 (이미 계산됨)
        # → patch features 만 따로 빼서 weak loss 계산
        
        student_patches_g1 = ...  # (B, N_patches, D)
        student_patches_g2 = ...
        
        patch_labels_g1 = data["patch_labels_g1"]  # (B, N_patches)
        patch_labels_g2 = data["patch_labels_g2"]
        
        weak_losses = []
        for b in range(student_patches_g1.shape[0]):
            for feats, labels in [
                (student_patches_g1[b], patch_labels_g1[b]),
                (student_patches_g2[b], patch_labels_g2[b]),
            ]:
                wl = per_patch_pairwise_loss(
                    feats, labels,
                    T=self.cfg.weak_sup.T,
                    background_class=self.cfg.weak_sup.background_class,
                    min_patches_per_class=self.cfg.weak_sup.min_patches_per_class,
                    skip_background=self.cfg.weak_sup.skip_background,
                )
                if wl.item() > 0:
                    weak_losses.append(wl)
        
        if weak_losses:
            L_weak = torch.stack(weak_losses).mean()
            # warmup
            lambda_w = self.cfg.weak_sup.lambda_W * min(
                iteration / max(self.cfg.weak_sup.warmup_steps, 1), 1.0
            )
            total_loss = total_loss + lambda_w * L_weak
            loss_dict["weak_loss"] = L_weak.detach()
            loss_dict["lambda_w"] = torch.tensor(lambda_w)
    
    # ... 기존 backward ...
```

### 3. Collate function for labeled batch

`patch_labels_g1`, `patch_labels_g2` 가 image 마다 다른 크기일 수 있어서 (
augmentation 결과에 따라) collate 시 stacking 처리 필요.

만약 같은 image_size 라면 동일 shape 이므로 default collate OK. 다만 meta
dict 등 처리 위해 custom collate 권장:

```python
def _labeled_collate(samples):
    return {
        "collated_global_crops": torch.stack([s["global_crops"][i] for s in samples for i in (0,1)]),
        "collated_local_crops": torch.stack([c for s in samples for c in s["local_crops"]]),
        "patch_labels_g1": torch.stack([s["patch_labels_g1"] for s in samples]),
        "patch_labels_g2": torch.stack([s["patch_labels_g2"] for s in samples]),
        "metas": [s["meta"] for s in samples],
    }
```

## 권장 진행 순서

1. **[현재] Skeleton 검토** — 신규 모듈 의도/구조 확인
2. **`labeled` module 단위 테스트** — `LabeledEMDataset` 생성, 한 image+mask 로드 확인
3. **`joint_transforms` 단위 테스트** — image+mask 동기 augmentation 시각화
4. **`patch_label` 단위 테스트** — mask → patch labels 변환 확인
5. **`losses` 단위 테스트** — toy input 으로 loss 값 확인 (예상 범위 내)
6. **`train.py` 통합** — `RatioMixedLoader` integration
7. **`ssl_meta_arch.py` 통합** — `forward_backward()` 의 weak_sup 분기
8. **Stage 1 sanity test** — `enabled: false` 로 기존과 동일 동작 확인
9. **Stage 2 sanity test** — toy labeled data 로 짧은 학습
10. **본격 Stage 1/2/3 학습**

## Toy Test 권장 — 통합 전

각 module 을 작은 toy input 으로 검증:

### `losses.py` 테스트

```python
import torch
from dinov3.train.weaksup import per_patch_pairwise_loss

# Synthetic: 100 patches, 768 dim, 3 classes (0=bg, 1, 2)
N, D = 100, 768
features = torch.randn(N, D)
labels = torch.cat([
    torch.zeros(30, dtype=torch.long),     # bg
    torch.ones(35, dtype=torch.long),       # class 1
    torch.full((35,), 2, dtype=torch.long), # class 2
])

# 가정 1: class 1, 2 가 random feature → sim 거의 0
loss = per_patch_pairwise_loss(features, labels, T=8.0)
print(f"Random features loss: {loss.item():.6f}  (예상: 매우 작음, 0.001 수준)")

# 가정 2: class 1, 2 가 인위적으로 비슷
features[labels == 1] = torch.randn(D)  # 같은 vector 다수 + noise
features[labels == 2] = features[labels == 1].clone() + 0.05 * torch.randn(D)
loss = per_patch_pairwise_loss(features, labels, T=8.0)
print(f"Similar features loss: {loss.item():.6f}  (예상: 큼, 0.3~0.6)")
```

## 주의 사항

### 통합 시 깨질 수 있는 부분

1. **FSDP wrap**: `LabeledImageAugmentation` 자체는 모델 아니라 무관. RatioMixedLoader 도 무관.
2. **Distributed sampler**: Labeled loader 는 단순 DataLoader 사용. 분산 환경에선 별도 처리 필요할 수 있음.
3. **Collate**: WebDataset 의 collate 결과 형식과 labeled collate 결과 형식이 일치해야 forward 가 동일하게 처리. `is_labeled` flag 로 분기하거나 형식 통일 필요.
4. **Mask 의 batch_size mismatch**: WebDataset batch 가 128, labeled batch 가 16 일 수 있음. SSLMetaArch forward 가 다른 batch size 도 지원하는지 확인.

### Stage 1 호환성 보장

`weak_sup.enabled: false` 일 때:
- `RatioMixedLoader` 자체를 생성하지 않음
- 기존 코드 경로 그대로 사용
- 동작이 기존 DINOv3 와 100% 동일해야 함 (sanity check 필수)

이 조건 만족하면 Stage 1 학습은 기존 DINOv3 코드와 차이 없습니다.

## 통합 후 검증 체크리스트

- [ ] Stage 1 config 로 학습 시작 → loss 곡선이 semi_MAE 의 v7 과 거의 일치
- [ ] Stage 2 config 로 학습 시작 → SSL loss + weak_loss 둘 다 logging
- [ ] `weak_loss` 가 시간 따라 감소 (self-curriculum 작동)
- [ ] `pair_sim_max` 가 점진 감소
- [ ] iBOT loss / DINO loss 가 발산 안 함 (weak sup 이 SSL 망치지 않음)
- [ ] gradient norm 이 clip 한계 안 (학습 안정)
- [ ] Stage 3 에서 weak_loss = 0 이 되지 않음 (preservation 작동)
