# Segmentation Tracking — Method 상세

`segmentation_tracking.ipynb` 가 쓰는 방법 정리. 연속 이미지(예: 표면 얼룩 확산)에서
첫 프레임 마스크를 나머지 프레임으로 전파(propagation)해 segmentation 한다.

## 0. 큰 그림 — "학습된 트래커"가 아니다

핵심은 **non-parametric label propagation** (Jabri et al. 2020, contrastive random walk 계열).
학습된 트래킹 헤드가 없다. **frozen DINO patch feature 의 유사도**만으로, 첫 프레임에 사람이 준
마스크를 다음 프레임들로 "복사·전파"한다. 즉 트래킹 로직 = **feature 매칭 + 공간/시간 제약**.

- backbone: domain-finetuned DINOv3 **ViT-B/16** (`setup_and_build_model` 로 로드)
- 트래킹 파라미터는 전부 하이퍼파라미터 (학습 불필요)

## 1. Feature pipeline (이미지 → patch feature)

```
PIL RGB
 └─ transform ─────────────────────────────────────────
    ResizeToMultiple(SHORT_SIDE=960, multiple=16)   # 짧은 변 960, patch 배수로 반올림
    ToTensor                                          # [3,H,W], 0..1
    PerImageNormalize                                 # per-image zero-mean/unit-std  ★
 └─ forward: DINO ViT-B ───────────────────────────────
    model.get_intermediate_layers(n=1, reshape=True, norm=True)
      → 마지막 블록 patch 토큰 [1, D, h, w]   (h=H/16, w=W/16, D=768)
    movedim → [h, w, D]
    F.normalize(dim=-1)                               # patch별 L2 정규화 → 코사인용
 → feats [h, w, D]
```

중요 포인트:
- **`PerImageNormalize`** (ImageNet 통계 아님): finetuned ViT-B 가 그렇게 학습됨. 틀리면 feature
  분포가 어긋나 매칭이 망가진다.
- **L2 정규화**: 이후 내적 = **코사인 유사도**. feature 크기가 아니라 방향(패턴)만 비교.
- 해상도 예: 960×1280 입력 → **60×80 feature grid**. 여기가 propagation 이 도는 "작업 해상도".

### intensity 채널 보강 (`INTENSITY_WEIGHT`)

**문제**: `PerImageNormalize` + DINO semantic feature 는 밝기/조명에 둔감하도록 설계됨. 그런데
액체 확산의 경계는 **미세한 밝기차**라, 정작 필요한 신호가 feature 에 약하게만 실린다. 그 결과
확산 front 가 기존 영역과 밝기만 조금 달라도 매칭에서 배경으로 넘어가 **front 가 선택 안 되는** 문제.

**해결**: `forward()` 에서 DINO feature `[h,w,D]` 에 **patch 별 평균 밝기 채널**을 concat 한 뒤 다시
L2 정규화 → 코사인 유사도가 밝기차를 반영. `INTENSITY_WEIGHT` 로 semantic 대비 밝기 비중 조절
(0=끔, 0.3~1.0 권장). 학습 불필요. 모든 프레임이 동일 `forward` 를 타므로 context/current 일관.

## 2. 첫 프레임 준비 (seed)

- **첫 마스크 PNG** (0=배경, 1,2,…=객체) 를 사람이 제공 → "무엇을 추적할지"의 정의.
- 마스크를 feature 해상도(60×80)로 nearest 다운샘플 → `first_mask [h,w]`
- one-hot → `first_probs [h,w,M]` (M = 배경 포함 라벨 수).
- `first_feats`, `first_probs` 는 **항상 context 에 고정 포함**되어 앵커 역할.

## 3. Propagation 로직 (`propagate()`) — 트래킹의 심장

입력: `current_feats[h,w,D]`, `context_feats[t,h,w,D]`, `context_probs[t,h,w,M]`.

1. **유사도**: `dot = einsum(current, context)` → `[h",w",t,h,w]`. 현재 patch ↔ 모든 context patch 코사인.
2. **공간 지역성 제약** (`neighborhood_mask`) ★ 정체성의 핵심
   현재 patch `(i,j)` 는 자기 위치 반경 `NEIGHBORHOOD_SIZE=12` patch(circle) 안의 context 하고만 매칭.
   밖은 `-inf`. → feature 가 비슷해도 멀리 있는 건 제외. "같은 자리에 있던 그 객체"라는 연속성 강제.
3. **top-k**: 이웃 안에서 가장 유사한 `TOPK=5` 개만 남김.
4. **가중 투표**: `weights = softmax(dot / TEMPERATURE)`, `TEMPERATURE=0.2`.
   `current_probs = weights @ context_probs` → 선택된 patch 라벨의 유사도 가중평균 → `[h",w",M]`.

> 요약: "내 주변 반경 안에서, feature 가 가장 닮은 top-5 context patch 들의 라벨을 유사도로 투표받아 내 라벨을 정한다."

| 파라미터 | 역할 | ↑ 하면 |
|---|---|---|
| `NEIGHBORHOOD_SIZE=12` | 매칭 허용 반경 | 큰 움직임 허용 / 인접 객체 번짐 ↑ |
| `TOPK=5` | 투표 참여 수 | 다수결 안정 / 소수 신호 희석 |
| `TEMPERATURE=0.2` | softmax 날카로움 | 부드럽게(민감도↓) / 낮추면 winner-take-all |

## 4. 시간 전파 (메인 루프) — context queue

```
context = [first_frame(고정)] + [최근 MAX_CONTEXT_LENGTH=7 프레임 queue]
for frame_idx in 1..N:
    current_feats = forward(frame)
    current_probs = propagate(current_feats, context_feats, context_probs, ...)   # 60×80
    queue += (current_feats, current_probs);  7 초과 시 오래된 것 pop
    probs_lowres[frame_idx] = current_probs           # 저해상 저장 (JBU 입력용)
    # 출력: nearest 업샘플 → postprocess_probs(per-mask min-max) → argmax
    mask_predictions[frame_idx] = argmax
```

- **정체성이 이어지는 이유**: 직전 프레임 라벨이 queue → 다음 프레임 context. 객체가 조금 움직여도
  이전 위치가 반경 안이라 라벨 계승.
- **first frame 고정 포함**: drift(누적 오차)를 앵커로 억제.
- ⚠️ 자가 교정 없음: 한번 틀리면 queue 를 통해 전파.

## 5. 출력 정제 (후처리 — tracking 에 영향 없음)

**propagation 루프는 손대지 않고** 출력 마스크만 다듬는다 (오차 피드백 누적 방지).

### (A) 형태학 prior → `mask_predictions_pp`
- `FILL_HOLES`: 영역 내부에 둘러싸인 구멍(=particle) 크기 무관 메움.
- `CLOSING_RADIUS`: R px disk closing (경계 걸친 이물 ~2R 까지). = "허용 particle 최대 크기" 노브.
- 목적: feature 가 particle 을 "정확히" 제외하는 과민함을 **영역 단위 상식**으로 보정.

### (B) JBU — Joint Bilateral Upsampling → `mask_predictions_jbu`
- 입력: 저해상 확률맵 `probs_lowres[t]` (60×80) + guide = 원본 RGB 프레임.
- 각 고해상 픽셀 p = 대응 저해상 이웃 q 들의 확률을 **spatial gaussian × range(색차) gaussian** 으로 합성
  → **이미지 엣지를 넘는 이웃은 배제** → 계단형 경계가 실제 경계에 밀착.
- 파라미터: `JBU_RADIUS`, `JBU_SIGMA_SPATIAL`(patch 단위), `JBU_SIGMA_RANGE`(색차, 작을수록 엣지 밀착),
  `APPLY_MORPH_ON_JBU`(JBU 후 형태학도 적용).
- 학습 불필요 (FeatUp 의 JBU 변형과 동일 원리).

시각화 셀은 실행한 것들(`raw`/`pp`/`jbu`)을 자동으로 나란히 비교.

## 6. 데이터 흐름 한눈에

```
frames[t] ─transform─→ [3,H,W] ─DINO─→ feats[60,80,768]
                                          │
first mask ─downsample→ first_probs ──────┤
                                          ▼
                    propagate(neighborhood 12 + top5 + softmax0.2)
                                          │  context = first + 최근7
                                          ▼
                          current_probs[60,80,M] ──→ probs_lowres[t]
                             │ nearest↑                    │
                             ▼                             ▼ (출력 정제)
                   mask_predictions[t]        pp: fill/closing   jbu: RGB-guide 업샘플
```

## 7. 특성 · 한계 (얼룩 확산 관점)

- **강점**: 단일 영역(얼룩)이라 인스턴스 교차 문제 없음. 첫 프레임에 얼룩 하나만 칠하면 커지는 경계 추종.
- **핵심 튜닝**: 얼룩이 프레임 간 반경(12 patch ≈ 192px @960)보다 빨리 번지면 놓침
  → `NEIGHBORHOOD_SIZE` ↑ 또는 `SHORT_SIDE` ↑.
- **경계 품질**: 60×80 에서 도는 한계 → **JBU 가 경계를, 형태학이 내부 particle 을** 각각 보정.
- **자가 교정 없음** → 초반 몇 프레임 결과를 꼭 확인 (drift 조기 감지).
- 여러 개의 **비슷한 인스턴스가 교차/겹치는** 경우엔 이 방법이 약함 (appearance 로 구분 불가,
  공간+시간 연속성에만 의존) → 그런 목적이면 SAM2 video / re-ID MOT 가 더 적합.

## 참고 — sliding window 미채택

feature 해상도를 올리는 shifted-window(입력 8px 이동 후 다중 forward interleave) 방식은 segment
tracking 에서 프레임별 grid 정렬/정합 위험이 있어 보류. 해상도가 더 필요하면 `SHORT_SIDE` 를 올리는
단순 방법을 우선.
