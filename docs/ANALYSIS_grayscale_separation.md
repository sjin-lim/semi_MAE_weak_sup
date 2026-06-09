# 왜 DINOv3 SSL은 "옅은 회색 차이 + 흐린 경계" 두 물체를 분리하기 어려운가

> EM 도메인에서 흑백으로 아주 약간 차이나는 두 물질(예: RGB 147 vs 168)이 흐린 경계를
> 맞대고 있을 때, SSL(DINOv3)이 이를 분리하는 표현을 학습하기 어려운 이유의 메커니즘 분석.
>
> **2026-06-09 개정**: PCA 실측에 근거해 중심 메커니즘을 *augmentation invariance*(이전 K1)에서
> **반사 variance에 의한 masking**으로 교정. gamma 등 광도 aug는 *주범*에서 *보조*로 강등.
>
> 강한 주장(메커니즘적으로 확실)과 가설(정황상 유력)을 구분해 기술.

---

## 0. 문제 설정 (수식화)

이미지 강도장 $I(p)$, 두 영역 $\Omega_A, \Omega_B$:

- $\mu_A \approx 147/255 \approx 0.576$, $\mu_B \approx 168/255 \approx 0.659$
- 클래스 신호 = DC 오프셋: $\Delta_{\text{mat}} = |\mu_A - \mu_B| \approx 21/255 \approx 0.082$
- 경계가 흐림 → 경계 gradient 작고 넓게 퍼짐, 영역 내부는 거의 flat
- **반사 artifact**: 금속 분포 간섭으로 생기는 **파동(ripple)** 밝기 변화. 진폭이 물질 gap보다
  훨씬 큼(수십~100 레벨). 물질 내부에도, 이미지 전역에도 나타남.

→ **물질을 가르는 신호는 저분산·저주파 1차원(DC). 반사는 고분산·구조적.** 이 비대칭이 핵심.

---

## 1. 중심 원리 (개정): 모델은 intensity에 *무감각*한 게 아니라, *엉뚱한 intensity(반사)에 지배*된다

PCA 실측 결과:
- 학습된(v7까지) feature의 **top principal component가 반사 밝기 그라데이션**을 따라감
  (동일 물질 내부가 반사 따라 다른 색으로 갈라짐).
- 정작 **물질 간 DC 차이는 뚜렷한 PC로 나타나지 않음** (묻힘).

→ 즉 모델은 "intensity invariant"가 **아니다**(그랬다면 반사에도 무반응이어야 함).
오히려 **반사라는 지배적 intensity 변동이 표현 capacity를 독식**해서, 미세한 물질 DC가
표현의 noise floor 아래로 가라앉는다. **"intensity 무감각"이 아니라 "intensity-dominated by the
wrong signal".** (이전 판의 K1 "intensity invariance" 서술은 이 PCA 증거와 모순 → 폐기/강등.)

---

## 2. 메커니즘 (재정렬)

### M1 (primary). 반사 variance가 capacity를 독식 → 물질 DC가 masking됨

학습된 표현의 capacity는 **고분산·구조적·augmentation-robust** 신호에 우선 배분된다
(contrastive/centering objective가 고분산 방향을 최대 활용):

$$
\max_\theta\; I(f_\theta;\, \text{content}_{\mathcal T\text{-inv}}) - \beta\, I(f_\theta; x)
$$

반사 wave는 **(고분산) + (공간 구조) + (단조 광도 aug에 robust)** 세 조건을 모두 만족 → top PC
독식. 물질 DC는 **(저분산) + (광도 aug가 noise 주입, M2)** → 같은 표현 안에서 경쟁에 밀려 폐기.

LayerNorm/instance_norm도 거든다: 토큰별 정규화가 패치의 DC 성분을 압축해 저분산 DC 축의
생존을 더 어렵게 함. (단 반사는 *여러 패치에 걸친 공간 coherence*로 살아남음 — §3 참조.)

> 결과적 증상: 물질 A·B는 **병합**(sim[1,2]↑), 물질 내부 반사는 **과분할**. 둘 다 "반사 dominant,
> 물질 DC 묻힘"이라는 하나의 원인에서 나옴.

### M2 (secondary). 광도 aug(gamma/CLAHE)는 *물질-DC 축에만* noise를 주입

이전 판은 이걸 주범(K1)으로 봤으나, PCA상 모델은 intensity-invariant가 아니므로 강등.
정확한 역할: 전역 단조 광도 aug $g_t$는 **절대 레벨(=물질 축)** 을 흔들지만($\mu\mapsto g_t(\mu)$),
**반사의 공간 구조**는 거의 못 흔든다($\nabla(g_t\circ I)=g_t'\nabla I$, 패턴 보존). 즉 gamma는
"모든 intensity를 무시하게" 만드는 게 아니라 **물질 DC 채널의 SNR만 추가로 깎는 보조 noise**다.

### M3. 흐린 경계 → edge cue 부재 + iBOT class-agnostic

경계가 흐리면 latch할 edge feature가 없고, 남는 단서 내부 DC는 M1/M2가 죽인다. iBOT 마스킹
예측도 flat 내부에선 class-agnostic:

$$
h(\text{masked }A_{\text{int}}) \approx h(\text{masked }B_{\text{int}}) \approx \text{"generic flat region"}
$$

> info-aware masking을 끈 v7 결정과 일치(interior 반복 마스킹 = A,B interior 똑같이 채우기).

---

## 3. 역설 = M1의 직접 증거: 물질 *내부* 반사는 나누고, 물질 *간*은 합친다

| | 사람 판단 | 모델 동작 | 이유(M1) |
|---|---|---|---|
| 물질 내부 반사 gradient | 같은 물질 (합쳐야) | **나눔** (PCA 그라데이션) | 반사 = 고분산·구조·aug-robust → top PC |
| 물질 A vs B DC 차이 | 다른 물질 (나눠야) | **합침** | 물질 DC = 저분산·DC-noise → 묻힘 |

왜 반사는 LayerNorm을 뚫고 살아남나 — 세 요인: **(1) 공간 coherence**(수십 패치의 일관된 패턴 →
attention 증폭), **(2) 큰 진폭**(물질 gap보다 큼 → 잔여 분산 큼), **(3) 동반 텍스처**(국소 대비·
specular 변화도 동반). 물질 gap은 셋 다 약함. → 보존 intensity 정보량 ≈ (분산)×(공간구조)×(aug-robust),
**반사는 셋 다 높고 물질은 셋 다 낮다.**

사람은 **lightness constancy(조명 discount) + material prior**를 함께 쓰지만, 모델은 후자가
전무 → 보이는 반사 구조를 그대로 feature 분산으로 옮기고 물질 경계는 놓침.

---

## 4. SNR 통합 — 분모의 지배항은 반사

$$
\mathrm{SNR}_{\text{material}} \;=\; \frac{\Delta_{\text{mat}}}
{\underbrace{\sigma_{\text{reflection}}}_{\text{지배항(M1)}} \;+\; \underbrace{\sigma_{\gamma}}_{\text{보조(M2)}} \;+\; \sigma_{\text{noise}}}
$$

$\sigma_{\text{reflection}} \gg \Delta_{\text{mat}}$이면 물질은 표현상 분리 불가. **분모를 줄이는
가장 큰 레버는 $\sigma_{\text{reflection}}$ 제거**(반사 invariance/제거)이지 $\sigma_\gamma$가 아니다.

---

## 5. augmentation 역할 재평가

- **gamma-off(config 반영됨)** = 분모의 *작은 항* $\sigma_\gamma$만 줄임 → **marginal일 수밖에 없음.**
  너희 v6 ablation이 marginal이었던 것과 정확히 일치(이전 "K1 강함" 설명보다 데이터에 더 부합).
- **여전히 gamma-off는 유지**(물질 DC 채널에 불필요한 noise 안 주는 게 맞음, 공짜·무해).
- **1순위 레버는 $\sigma_{\text{reflection}}$ 제거** → 아래 §6.

(geometric/masking aug는 SSL 핵심이라 그대로 유지.)

---

## 6. "이 빛 변화가 noise"라고 모델에 알려주는 방법

"X는 noise"를 가르친다 = **X에 invariant하게 만든다.** invariance 신호를 *어떻게 공급하느냐*로
분류:

| 공급 방식 | 전제 | 방법 |
|---|---|---|
| **생성(generate)** | X를 합성 가능 | wave aug (student-only) → consistency가 invariance 학습 |
| **제거(remove)** | X를 분리 가능 | FFT notch / homomorphic de-wave (입력 단계 제거) |
| **측정(measure)** | X의 proxy 측정 가능 | feature와 wave-energy의 **decorrelation 페널티** |
| **탐지(detect)** | X 생성·제거 불가, 탐지만 | **adversarial(gradient reversal)** head |
| **라벨(label)** | 무엇이 같아야 하는지 앎 | semi-sup intra-class invariance |

### 우선순위 (단순·정합 순)

1. **wave aug(student-only) + 기존 DINO consistency** — *새 head 불필요.*
   labeled/unlabeled student view에만 wave를 주면, teacher(미적용) ↔ student(waved) 사이의
   **DINO/iBOT consistency가 "wave=noise"를 자동 학습**한다. weak push가 "material=signal"을
   담당 → **두 기존 메커니즘이 분리 학습**(§7 Pareto 회피). 너가 aux-loss 충돌을 싫어한 점과도 부합.
2. **de-wave 전처리(FFT notch)** — 모델 무변경, 데이터에서 $\sigma_{\text{reflection}}$ 직접 제거.
   band가 깨끗이 분리되면 강력. 단 구조 edge와 겹치면 손상 → 검증 필수. wave 있는 이미지의
   *clean teacher anchor* 생성용으로도 활용(upgrade).
3. **decorrelation 페널티** — 패치별 wave-energy(국소 band-pass 에너지)를 측정해, feature와의
   상관/HSIC을 최소화. 경량 통계적 "이 축은 noise". aug로 부족할 때 보강.
4. **adversarial nuisance head (gradient reversal)** — backbone이 wave를 *예측하지 못하도록*
   적대적으로 페널티 → 표현에서 wave 정보 제거. 가장 직접적이나 **학습 불안정·loss 충돌 위험**
   (네가 기피한 aux-head 계열) → 1·2로 부족할 때만.

> 핵심: **대부분의 효과는 1번(wave aug)으로 새 loss 없이 얻을 수 있다.** "noise라고 알려주기"가
> 곧 "그 변형에 대한 consistency 강제"이고, DINO는 이미 consistency 기계이기 때문.

(실측 band 추정·합성 검증: `notebooks/wave_artifact_analysis.ipynb`.)

---

## 7. 왜 semi-supervised가 *선택이 아니라 필수*인가 (그리고 경험적으로 확인됨)

> 실측상 semi-sup가 명확히 효과를 보임. 아래는 *왜 원리적으로* 그래야 하는지.

### 7.1 정보 이론 — 없는 비트는 unlabeled로 안 생긴다

물질 정체성 $Y$는 절대 레벨의 함수인데, SSL이 추출하는 통계(aug-불변 ∩ 고분산)와 거의 직교:

$$
I\big(Y;\ \phi(x)\big) \approx 0 \quad \text{for } \phi \in \{\text{aug-invariant} \cap \text{high-variance}\}
$$

→ **샘플 수 문제가 아님.** unlabeled를 100배 모아도 추출 가능 통계가 $Y$와 직교하면 정보량은
0. 소량 label이 이 비트를 직접 공급 → label efficiency가 본질적으로 높음.

### 7.2 confound = Pareto ceiling

반사와 물질이 **같은 "intensity 민감도" 손잡이**에 엉켜 있어, SSL knob은 무엇을 당겨도
(반사 과분할 ↔ 물질 병합) front 위를 미끄러질 뿐이다. **front를 벗어나려면 "이 강도변화는 조명,
저것은 물질"을 알려주는 외부 축(material prior)** 이 필요. 그 prior는 어떤 augmentation으로도
못 만든다 — **오직 label(또는 근사 region prior)만** confound를 분리.

> §6의 wave-aug와의 시너지: wave-aug가 *within-region 반사*(한 손잡이)를, weak push가
> *between-region 물질*(다른 손잡이)을 따로 담당 → **서로 다른 구조를 키로 써서 confound를
> 부분적으로 분리.** 둘이 상보, 그리고 wave-aug는 weak sup의 enabler(반사 masking을 걷어
> per-patch 신호를 표면화).

### 7.3 왜 full supervised가 아니라 weak/semi인가

1. **label 희소**(수천 장) → full CE는 overfit, SSL 일반표현 덮어씀
2. **고정 class head 불가** (dataset마다 class 수 다름)
3. **SSL 구조 파괴** ("패치→자기 class로" 강한 pull → prototype 붕괴)
4. 필요한 건 표현 전체가 아니라 **disentangling 축 하나** → per-patch pair-wise push(label 있는
   곳만)로 최소 침습 주입이면 충분

→ **semi-sup = SSL이 못 만드는 material prior 비트를, 표현을 안 망치고 가장 label-efficient하게
주입.** full은 과하고(파괴), pure SSL은 모자라다(축 부재).

---

## 8. 검증 제안

1. **wave-aug 후 PCA 재측정**: top PC에서 반사 그라데이션이 줄고, 물질 분리 방향이 올라오나
   (M1 교정의 직접 검증 — 가장 중요).
2. **$\sigma_{\text{reflection}}$ vs $\Delta_{\text{mat}}$ 측정**: FFT band 에너지로 반사 분산 정량
   (`notebooks/wave_artifact_analysis.ipynb`).
3. **층별 Fisher ratio** $\dfrac{(\bar f_A-\bar f_B)^2}{\sigma_A^2+\sigma_B^2}$ — 깊을수록 하락?
4. **축 정렬**: 반사 PCA 방향 vs 물질 LDA 방향 $\cos\angle$ (직교면 push-only 안전).

---

## 9. 정리 표

| | 메커니즘 | 비중(개정) | 기존 관찰 연결 |
|---|---|---|---|
| **M1** | 반사 variance가 capacity 독식 → 물질 DC masking | **주(확실, PCA)** | PCA 반사 그라데이션, sim[1,2] 병합 |
| **M2** | gamma 등 광도 aug = 물질-DC 축 noise | 보조(강등) | v6 aug ablation marginal |
| **M3** | 흐린 경계 = edge 부재 + iBOT class-agnostic | 중간 | info-aware masking harmful |
| 상한 | material prior 부재 → 병합↔과분할 Pareto | — | pure SSL 한계, semi-sup 효과 확인 |

**한 줄 결론**: 범인은 "intensity invariance"가 아니라 **반사 variance가 물질 DC를 masking**하는
것. 1순위 레버는 **반사 제거/invariance**(wave-aug, de-wave; §6), 천장은 **semi-sup(material
prior)**로 뚫는다(§7). gamma-off는 공짜 보조일 뿐.

(weak loss 설계: [HANDOFF_CONTEXT.md](../HANDOFF_CONTEXT.md) §4, 구현:
[losses.py](../dino_v3/dinov3/train/weaksup/losses.py), wave 도구:
[wave_artifact_analysis.ipynb](../notebooks/wave_artifact_analysis.ipynb).)
