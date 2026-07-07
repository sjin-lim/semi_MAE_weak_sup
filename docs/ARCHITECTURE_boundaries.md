# 아키텍처 경계 (backbone ↔ downstream)

레포의 의존 계층과 지켜야 할 방향, 그리고 **알려진 결합(known coupling)**을 기록한다.

## 계층과 의존 방향

```
util/ (semi_MAE 데이터 유틸)        ← dino_v3/dinov3 (백본: SSL 학습·모델·eval)
                                          ▲
                                          │  inspection → dinov3 (OK)
                                    inspection/ (downstream 제품: 분류·측정·서비스)
```

**규칙:**
- `inspection → dinov3` 만 허용(단방향). 백본은 downstream 을 몰라야 한다.
- `dinov3` 는 `inspection`(및 어떤 downstream 태스크)도 **import 하지 않는다.**

**검증 (2026-07 기준):**
- `dinov3 → inspection` import: **0건** ✓ (백본이 우리 분류/서비스 코드에 의존하지 않음)
- `util → dinov3` 역참조: **없음** ✓ (순환 없음)
- `hub/{segmentors,detectors,classifiers}` + `eval/{segmentation,detection}` 은 **Meta DINOv3 원본**이고
  `dinov3` 패키지 **내부**에서만 참조(self-contained). 외부 downstream 결합 아님.

## 알려진 결합 (수용 중)

**`dinov3 → repo-root util.dataset_csv`** — 학습/데이터 로딩 3개 파일이 sys.path 로 repo 루트를
참조한다:
- `dino_v3/dinov3/data/augmentations.py` → `PerImageNormalize`
- `dino_v3/dinov3/data/datasets/h5_dataset.py` → `CSVImageDataset`, `load_dataframe`
- `dino_v3/dinov3/data/labeled/joint_transforms.py` → `PerImageNormalize`

**성격:** downstream **태스크** 의존이 아니라 semi_MAE **학습 데이터 유틸**(H5/CSV 로딩, per-image
정규화)에 대한 SSL 학습-시점 결합. 사용자 원칙("백본은 downstream 에 의존 금지")을 위반하지 않는다.

**현재 판단:** `util/` 이 `dino_v3/` 와 항상 함께 다니므로 **그대로 둔다.** 굳이 지금 풀 필요 없음.

**단, 인지할 것:** `dinov3` 는 이 결합 때문에 **완전 self-contained(단독 vendor) 는 아니다** —
`util/` 이 repo 루트에 있어야 학습이 돈다. 만약 나중에 **dinov3 를 독립 라이브러리로 떼어낼**
경우엔 위 primitive(`PerImageNormalize`, CSV/H5 로더)를 `util/` 에서 `dinov3/data/` 안으로
이관해 방향을 정상화해야 한다.

> 참고: `inspection/em_aug.py` 는 자체 `PerImageNormalize` 복사본을 갖는다(백본과 분리 목적).
> 위 이관을 하게 되면 중복 정리도 함께 고려.
