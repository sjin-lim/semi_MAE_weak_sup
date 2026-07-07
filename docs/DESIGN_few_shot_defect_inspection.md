# DINO 기반 Few-shot 불량 검사 설계서 (draft)

> 상태: 논의 정리용 초안. 피드백 반영 전. 구현 시작 전 합의 목적.

## 1. 목적

DINOv3(weak-sup) foundation 모델을 **공통 feature 추출기**로 두고, **소수의 불량
이미지**만으로 불량을 검출·분류한다. 백본은 재학습하지 않고(frozen) feature 위에
가벼운 헤드/측정 로직만 얹는다. 필요 시 **rough mask/bbox** 제작은 가능하다는 전제.ㅡㅡㅡㅡ

이미지 도메인: 규칙적으로 배열된 작은 홀(hole)을 가진 고해상도 EM 이미지.

## 2. 요구사항

- 소수(few-shot) 불량 이미지로 불량 **종류 구별(classification)**.
- DINO foundation 모델 활용(frozen, 재학습 없음).
- 필요 시 rough mask 제작 가능(픽셀 완벽 불필요).

## 3. 불량 종류와 공간 특성(footprint)

설계의 핵심 축은 "불량이 이미지에서 차지하는 공간적 형태"다. 이게 방법을 결정한다.

| # | 불량 | 공간 특성(footprint) | 판별 근거 |
|---|---|---|---|
| 1 | **전역 얼룩** (경계선 생기고 전체 영향) | 이미지 전반에 분산 | appearance(전역 통계 shift) |
| 2 | **홀 사이 뜯김 / 파티클** | 국소·희소, 곳곳에 분산 | appearance(국소 패턴) |
| 3 | **홀 내부 검은 점 과대** | 홀 단위 국소 | **크기(geometry)** — appearance 아님 |

**핵심 통찰:** 전역 image-level 분류가 **1차로 상당 부분을 커버**한다 — 경험적으로 간단한
분류 모듈이 particle 불량 이미지도 검출했다(국소 이상도 전역 통계를 흔들기 때문, 특히
max/std pooling에서). 따라서 세 방법을 병렬로 다 만들지 않고, **단순한 전역 분류를 1차
게이트로 두고 한계가 드러나는 부분만 정밀 모듈로 확장**한다(캐스케이드).
- 1번(얼룩): 전역 분류로 충분할 가능성 높음.
- 2번(파티클): 전역 분류로 "검출"은 됨. 정밀 위치/종류구별이 요구될 때만 patch-level(mask) 보강.
- 3번(검은점 크기): 정상 홀에도 검은 중심이 있어 **크기만 다름** → 전역/appearance 분류가
  가장 약한 지점. 필요 시 기하 측정으로 보강.

## 4. 설계 원칙

1. **Frozen backbone.** 파라미터 업데이트 없음. few-shot 과적합/OOM 회피.
2. **Footprint에 방법을 맞춘다.** 전역→image-level, 국소→patch-level, 크기→측정.
3. **해상도 보존.** 448로 다운샘플하지 않는다. 미세 파티클이 사라지기 때문.
   본 체크포인트는 stage3 mixed-res(512/768) + upscale(768→1024→2048) 적응이 되어 있어
   고해상도 직접 추론이 적응 범위 안(RoPE, 고정 pos-embed 없음).
4. **Mask는 최소·저비용.** 픽셀 완벽 불필요. few-shot 예시에만 rough하게. 학습용(트랙2)과
   보정용(트랙3)으로 역할이 다름.
5. **정규화 일치.** 학습이 `instance_norm`(PerImageNormalize)이므로 추론도 동일(현 도구 내장).
6. **단순부터, 데이터가 증명할 때만 확장(캐스케이드).** 전역 분류를 1차 게이트로 먼저 세우고
   separability로 실측 → 부족한 지점에만 정밀 모듈(patch-level·측정)을 추가. 세 방법을 미리
   다 만들지 않는다.
7. **Stage 2는 플러그인 프레임워크.** 측정은 첫 모듈일 뿐, 다양한 분석이 같은 인터페이스로
   꽂히도록 확장성 있게 설계(공통 컨텍스트 → 구조화된 결과).

## 5. 아키텍처 — 단계형 캐스케이드 (Staged Cascade)

전역 분류를 **1차 게이트**로, 정밀 모듈은 **필요 시 on-demand**. 대부분 이미지는 Stage 1에서
끝나고, 애매하거나 정밀도가 요구되는 경우만 Stage 2로 내려간다.

```
이미지 ─▶ [Stage 1: 전역 image-level 분류] ─── 정상 ──▶ PASS
                    │
             불량/저confidence/정밀요구
                    ▼
          [Stage 2: 필요 모듈만 on-demand]
            2a 위치설명(heatmap) · 2b patch-level 종류(mask) · 2c 검은점 크기측정
```

### 5.0 공통 기반 (Stage 1이 먼저 필요로 하는 것)
- **(a) 해상도 native화**: 448 강제 resize 제거, 작업 해상도 설정화(예: 1024, patch 16 배수).
- **(b) rich pooling**: image-level용 `[CLS ‖ mean ‖ max ‖ std(또는 top-k)]`.
  `mean`=전역 shift(얼룩), `max`/`top-k`=국소 이상(파티클), `std`=patch 이질성.
- (c) **patch-grid 출력**(H/16 × W/16 × D)은 Stage 2가 생길 때 추가. Stage 1엔 불필요.

### Stage 1 — 전역 image-level 분류 (1차 게이트, 항상)
- rich-pooled feature 위에 `DefectRegistry`(클래스별 feature 캐시) + logreg/NCM 헤드.
  **분류기 하나.** 소수 예시로 클래스 추가/재구성(백본 forward 없이 즉시). **mask 불필요.**
- 출력: 정상 / 불량종류(얼룩·파티clesize·검은점) + confidence.
- 경험상 particle 불량도 여기서 검출됨 → **1차 불량 가능성 분리**가 이 단계의 핵심 목적.
- 대다수를 여기서 처리하고 종료. **이것이 MVP.**

### Stage 2 — 확장 가능한 분석 모듈 프레임워크 (Stage 1 검출 이후)
Stage 2는 "측정" 하나로 고정하지 않고, **여러 분석 모듈이 꽂히는 프레임워크**로 설계한다.
정량 측정은 첫 번째 범주일 뿐, 향후 다른 계산(균일도, 경계 길이, 결함 밀도, 추세 등)이
**같은 인터페이스로 추가**될 수 있어야 한다(확장성 우선).

**공통 컨텍스트(모듈 입력):** 원본 이미지 · 고해상 patch-grid feature · Stage 1 분류결과 ·
홀 격자 정보 · 정상 기준(reference) 통계.
**모듈 출력(구조화):** 스칼라 지표 · 영역 mask · 홀별 값 · 시각화.
모듈은 이름/적용 불량종류로 등록되고, Stage 1 class가 해당 모듈로 라우팅된다(플러그인 패턴,
`DefectRegistry`와 동일 철학).

실행 트리거: (i) **정량/분석 출력이 요구될 때**(분류가 확신해도 실행), (ii) 크기 경계처럼
분류가 판정 못 하는 축.

**측정 모듈 (첫 범주, 면적은 px 단위 — 단위 변환은 후처리):**
- **얼룩 면적/경계**: `region_segmentation_explore` 방식이 얼룩에 더 적합(초기 실험 기준).
  `fewshot_heatmap`(reference-subtraction)도 가능하나 성능이 다소 낮음 → **방법은 추후 비교 확정**.
- **검은점 크기**: 홀 격자 검출 → 홀별 어두운 blob **면적(px)** + 임계(집단 outlier).
  appearance 분류가 못 하는 유일 축. mask는 **임계 보정용**(학습 아님).
- **파티클/뜯김 위치·개수**: (필요 시) few-shot **rough mask** → patch prototype 분류.
  mask→patch label은 weak-sup 재사용.

공통: 측정은 **보정 임계 + 정상 기준(reference set)** 필요(→ §8.1).

**결정 주체:** Stage 1 분류기 = 불량종류 결정(대다수 여기서 끝). Stage 2 모듈 = 지표/분석
산출 + 보정 규칙으로 정량 판정. **새 분석은 모듈 추가로 확장.** VLM은 코어 루프 밖(옵션, §8).

## 6. 기존 자산 재사용

| 자산 | 용도 |
|---|---|
| `inspection/service/feature_service.py` | 백본 feature 서비스. `include=patch`(그리드 반환) + 해상도 옵션 확장 |
| `inspection/em_classifier.py` (`EMFeatureExtractor`, `pool_tokens`, `DefectRegistry`) | Stage 1 골격 + rich pooling 추가 |
| `inspection/fewshot_separability.py` | 각 불량종류/해상도/pooling 분리도 사전 검증 |
| `notebooks/region_segmentation_explore.ipynb` | **얼룩 면적 측정 유력 방법**(heatmap 대비 우세) + 홀 격자 검출 groundwork |
| `inspection/fewshot_heatmap.py` (reference-subtraction) | 위치 설명(XAI). 얼룩 면적 대안(성능 다소 낮음) |
| weak-sup `mask_to_patch_labels`, joint_transforms, labeled dataset | 파티클 patch prototype 구성 |

## 7. 설계 근거 (왜 대안이 부족한가)

- **제미나이의 자기참조(self-reference) 앵커링은 분산형 불량에 원리적 결함.** query 패치
  평균을 정상 기준으로 삼는데, **전면 얼룩이면 기준 자체가 오염**되어 대비가 상쇄된다.
  분산형은 자기참조 대신 **별도 정상 집합**을 기준으로 써야 한다.
- **image-level 분류는 국소 불량 "검출"엔 되지만, 정밀 종류구별·위치엔 한계.** 경험적으로
  particle 불량 이미지는 검출됨(전역 통계 흔듦). 다만 수천 patch 중 소수 이상에 image 라벨
  1개라 SNR이 낮아, **정밀** 종류구별/위치가 필요하면 patch 단위 supervision(mask)이 유리.
  → 그래서 Stage 1(검출)로 먼저 거르고, 정밀이 필요할 때만 Stage 2(mask)로 확장.
- **검은 점 크기는 appearance 방법 전부 부적합.** 크기는 feature 방향에 불변 →
  유사도/prototype으로 분리 불가. 기하 측정만 유효.

## 8. 열린 질문 (결정 필요)

1. **정상 이미지 세트**가 별도로 있나? (트랙1 정상 기준, 트랙2/3 reference·보정에 유용)
2. **작업 해상도** 타깃: 1024 권장(적응 대역). 2048까지 필요/가능한가? (VRAM·속도 trade-off)
3. **측정 출력** — 면적은 **px 기준 산출**(단위 변환은 후처리). 얼룩 면적·검은점 크기 요구됨.
   불량별 필요 지표(면적/경계/개수 등)와 요구 정밀도, 그리고 **향후 추가될 분석 종류**가
   있으면 미리 공유(Stage 2 모듈 인터페이스 설계에 반영).
4. 트랙 3 **검은 점 크기 임계**의 단위/기준(px, μm, 상대 비율)?
5. 실시간성/처리량 요구(장당 지연, 배치 크기)?
6. 트랙 2 불량 종류 수와 예시 장수(few-shot 규모)?

## 9. 실행 로드맵 (캐스케이드 — 단순부터)

- **P0 (공통 기반, Stage 1용):** feature 추출 해상도 native화 + rich pooling. (patch-grid는 P2 때)
- **P1 (Stage 1 = MVP):** separability로 정상/불량종류 분리 확인 → `DefectRegistry` 다중분류.
  **여기까지가 1차 제품.** 분류기 하나, mask 없음.
- **P2 (Stage 2 측정 — 정량 출력이 요구되면 착수, 불확실성과 무관):**
  - **얼룩 면적**: `fewshot_heatmap` reference 모드 → 임계 → 연결영역 면적/경계. **저비용(재사용).**
  - **검은점 크기**: 홀 격자 검출 → blob 면적 + outlier.
  - 둘 다 정상 기준(reference)과 임계 보정 필요.
- **P3 (Stage 2 정밀 종류/위치, 필요 시):** 파티클 등 국소 불량의 정밀 위치/개수가 부족할 때만.
  rough mask few-shot → patch prototype.
- 원칙: 측정(P2)은 **필요한 출력**에 따라, 정밀 종류구별(P3)은 **Stage 1 실측으로 부족이 증명될 때** 착수.

## 10. 검증

- Stage 1 **hold-out 정확도**(few-shot 에피소드 평균±std) + 혼동행렬 + 오탐/미탐율.
- Stage 2는 **위치/면적을 시각화**해 사람이 근거 확인(XAI).
- 해상도·pooling 조합을 separability로 비교해 Stage 1 최적 설정 도출.
- 서버 feature ↔ 로컬 feature 일관성 스팟 체크.
