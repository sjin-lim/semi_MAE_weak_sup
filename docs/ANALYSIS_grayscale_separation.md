# 왜 DINOv3 SSL은 "옅은 회색 차이 + 흐린 경계" 두 물체를 분리하기 어려운가

> EM 도메인에서 흑백으로 아주 약간 차이나는 두 물질(예: RGB 147 vs 168)이 흐린 경계를
> 맞대고 있을 때, SSL(DINOv3)이 이를 분리하는 표현을 학습하기 어려운 이유의 메커니즘 분석.
> + 역설(물질 내부 반사는 오히려 나눔) 해소, augmentation의 역할, **그리고 label 없이(SSL만)
> 개선하는 아이디어**.
>
> 강한 주장(메커니즘적으로 확실)과 가설(정황상 유력)을 구분해 기술.

---

## 0. 문제 설정 (수식화)

이미지 강도장 $I(p)$, 두 영역 $\Omega_A, \Omega_B$:

- $\mu_A \approx 147/255 \approx 0.576$, $\mu_B \approx 168/255 \approx 0.659$
- 클래스 신호 = DC 오프셋: $\Delta = |\mu_A - \mu_B| \approx 21/255 \approx 0.082$
- 경계가 흐림 → 경계 gradient $\lVert\nabla I\rVert$ 작고 넓게 퍼짐
- 영역 내부는 거의 flat (저텍스처)

즉 **두 클래스를 가르는 정보는 본질적으로 1차원(평균 밝기)이고, 저분산·저주파·저SNR**.
이것이 모든 문제의 뿌리.

DINO/iBOT objective를 도식화하면:

$$
\mathcal{L}_{\text{SSL}}(\theta) = \mathbb{E}_x\, \mathbb{E}_{t,t'\sim\mathcal{T}}\;
D\big(h(f_\theta(t\,x)),\; \mathrm{sg}[\,h(f_{\bar\theta}(t'\,x))\,]\big)
$$

수렴점에서 $f$는 (i) augmentation 군 $\mathcal{T}$에 **불변** $f(t\,x)\approx f(x)$, (ii)
capacity/centering 제약 하에서 **$\mathcal{T}$-불변 변동에 대해서만 정보적**이 된다.

---

## 1. 중심 원리: 모델은 "절대 레벨"을 버리고 "구조적/상대적 변화"를 살린다

핵심 통찰부터. 모델이 키로 삼는 건 intensity *자체*가 아니라 그 intensity의 **속성**이다.
augmentation $g_t$는 **전역 단조(monotone) 밝기 재매핑**(gamma/brightness/CLAHE)이므로:

**(a) 절대 레벨** $\mathcal{A}[I]=\mathrm{mean}(I)=\mu$

$$g_t \text{ 적용 후}: \quad \mu \;\mapsto\; g_t(\mu) \qquad (\text{aug-fragile})$$

**(b) 공간 gradient / 국소 대비** $\mathcal{G}[I]=\nabla I$

$$g_t \text{ 적용 후}: \quad \nabla(g_t \circ I) = g_t'(I)\,\nabla I \qquad (g_t'>0 \Rightarrow \text{부호·패턴 보존, aug-robust})$$

→ 모델은 "intensity invariance"가 아니라 **"절대 레벨은 discount, 공간 구조는 보존"**을 배운다.
이 한 문장이 이후 모든 현상(K1~K3 + 역설)을 설명한다.

---

## 2. 세 가지 킬러 (절대 레벨 신호가 죽는 경로)

### K1. Augmentation invariance가 클래스 차이를 "같은 것"으로 학습 (가장 강한 원인)

분리가 가능하려면 **aug가 만드는 intra-class 밝기 변동 < 클래스 간 gap**이어야 한다:

$$
\underbrace{\mathrm{range}_{t\sim\mathcal{T}}\,\lvert g_t(\mu) - \mu\rvert}_{\text{aug 밝기 섭동}}
\;<\;
\underbrace{\lvert\mu_A - \mu_B\rvert}_{\Delta \approx 0.082}
$$

gamma $\in [0.7,1.4]$만 돼도 $\mu=0.58$에서 $0.58^{0.7}=0.68,\ 0.58^{1.4}=0.46$ → 섭동 폭
**0.2 이상**. $g_t(\mu_A)$가 $\mu_B$를 뛰어넘는다 → "A를 gamma로 흔든 것" ≈ "원래 B" →
invariance가 **$f(A)\approx f(B)$를 강제**.

> 관찰 "CLAHE off marginal", "invariance 한계"와 일치. CLAHE/contrast 재매핑 = 같은 K1.

### K2. LayerNorm + 저분산 → 판별 축이 표현에서 가장 먼저 폐기

augmentation을 다 꺼도 남는 문제. **판별 축(DC 레벨)이 LayerNorm null space 근처에 있다.**

flat 패치(강도 $c$) patch-embed: $v(c) = c\,u + b,\ u=W\mathbf 1_{\text{patch}}$. Pre-LN:

$$
\mathrm{LN}(v(c)) = \frac{c(u-\bar u\mathbf 1) + (b-\bar b\mathbf 1)}
{\big\lVert c(u-\bar u\mathbf 1) + (b-\bar b\mathbf 1)\big\rVert}\sqrt{D}
\;\xrightarrow[c\ \text{large}]{}\;
\frac{u-\bar u\mathbf 1}{\lVert u-\bar u\mathbf 1\rVert}\sqrt D \quad(\text{c 무관})
$$

가까운 두 $c$에 대해 감도 $\partial\,\mathrm{LN}/\partial c$가 매우 작다. LN이 평균(DC)을 제거·정규화
→ **밝기 신호를 구조적으로 압축**. instance_norm $(x-\mu_{img})/\sigma_{img}$도 동일 방향.

여기에 centering(Sinkhorn) + $K=32768$ 프로토타입 균형 할당이 겹친다. 정보병목:

$$
\max_\theta\; I(f_\theta;\, \text{content}_{\mathcal T\text{-inv}}) - \beta\, I(f_\theta; x)
$$

A/B 방향은 **(a) 저분산 + (b) 저 $\mathcal T$-불변성**(gamma가 정확히 흔드는 축) → 압축적·불변추구
objective가 가장 먼저 버리는 두 성질을 모두 가짐.

> → ANALYSIS_0527의 "**L3→L11 깊을수록 sim[1,2] 악화**"가 이걸로 설명됨.

### K3. 흐린 경계 → edge cue 부재, iBOT reconstruction이 class-agnostic

경계가 흐리면 $\lVert\nabla I\rVert$ 작고 퍼져 latch할 edge feature 없음. 남는 단서 내부 DC는
K1/K2가 죽임. iBOT 마스킹 예측도:

$$
h(\text{masked }A_{\text{int}}) \approx h(\text{masked }B_{\text{int}})
\approx \text{"generic flat region" prototype}
$$

→ A/B 가르는 gradient $\approx 0$.

> info-aware masking을 끈 v7 결정과 연결: interior 반복 마스킹 = "A,B interior 똑같이 채우기"
> 학습 → 병합 강화. 랜덤 마스킹이 낫지만 흐린 경계에선 그 효과도 약함.

---

## 3. 역설: 물질 *내부* 반사 gradient는 오히려 나눈다 (중심 원리의 증거)

관찰: 동일 물질 내 반사로 생긴 intensity gradient는 PCA에서 **다른 색 그라데이션**으로 나타나
모델이 굳이 feature를 나눈다. 사람 눈엔 같은 물질인데도.

| | 사람 판단 | 모델 동작 |
|---|---|---|
| 물질 내부 반사 gradient | 같은 물질 (합쳐야) | **나눔** (PCA 그라데이션) |
| 물질 A vs B DC 차이 | 다른 물질 (나눠야) | **합침** |

**해소** (= §1 원리의 직접 귀결): 반사 ramp는 **(b) 공간 구조**라 aug-robust·고분산·공간 coherent →
살아남아 PCA에 뜸. 물질 gap은 **(a) 절대 레벨**이라 aug-fragile·저분산 → 폐기.

왜 LN/K2가 반사는 안 죽이나 — 세 요인:

1. **공간 coherence**: LN은 토큰별 정규화. 한 패치 DC는 압축하지만 수십 패치의 단조 ordering은
   토큰 간 *상대* 차이로 남아 attention이 증폭. 두 flat 영역은 공간 ordering 없이 패치별 DC만
   달라 압축되면 끝.
2. **분산 크기**: 반사 swing은 보통 물질 gap(21)보다 큼(50~100레벨) → 잔여 분산이 커 주성분 생존.
3. **동반 텍스처**: 반사는 국소 대비·specular·국소 통계도 바꿈(구조적·aug-robust). A/B는 텍스처
   동일·DC만 차이.

> 모델이 보존하는 intensity 정보량 $\approx$ (분산) × (공간 구조) × (aug-robustness).
> **반사는 셋 다 높고, 미세 물질 gap은 셋 다 낮다.** 같은 메커니즘, 정반대 결과.

### 아이러니 + 사람 눈과의 대비

모델은 K1으로 **일종의 illumination invariance**(절대 레벨 discount)를 배웠다. 그런데 두 물질을
가르는 유일 신호가 *바로 그 절대 레벨 차이*라, 모델엔 "조명이라 어두운 곳"과 "다른 물질이라 어두운
곳"이 **구별 불가**. 이 ambiguity를 task에 최악으로 해소:

- 절대 레벨 차이(물질) → 조명으로 간주 → discount → **합침**
- 구조적 gradient(반사) → 실재 구조로 간주 → 보존 → **나눔**

사람은 **(1) lightness constancy(조명 discount) + (2) material prior**("한 물질에 빛")를 함께 쓴다.
모델은 (1)만 (과하게) 배우고 **(2) material prior가 전무** → 보이는 구조 변화를 그대로 feature
분산으로 옮기며 over-segment, 21레벨 gap이 물질 경계인지 또 다른 조명 ripple인지 알 길 없음.
**모델의 invariance가 augmentation에 맞춰져 있지 material semantics에 맞춰져 있지 않다**는 게 본질.

---

## 4. SNR 통합 지표

$$
\mathrm{SNR} \;\approx\; \frac{\Delta}{\sigma_{\text{aug}}}
= \frac{\lvert\mu_A-\mu_B\rvert}{\sqrt{\sigma_{\gamma}^2+\sigma_{\text{poisson}}^2+\sigma_{\text{blur}}^2+\cdots}}
$$

$\sigma_{\text{aug}} \gtrsim \Delta$이면 patch 레벨에서 통계적 분리 불가. 현재 aug는 거의 이 영역.

---

## 5. augmentation이 문제를 키우는가? (Yes, 단 단서 있음)

**그렇다.** 현재 intensity aug(gamma/poisson/blur, 과거 CLAHE)는 K1을 직접 구현하고, blur는 K3
(경계 흐림·A/B 혼합)을 악화. 즉 aug는 **이 특정 문제를 키우는 방향**.

**그러나 단서** — 너희의 기존 negative result가 핵심을 말해준다:

- v6 aug ablation(CLAHE off 등) → marginal
- fixed-norm(절대 레벨 보존 시도) → 도움 안 됨

이유: **aug를 끄는 것은 K1(파괴)만 줄일 뿐 K2(저분산 축을 압축에서 폐기)는 그대로**. 신호를
"덜 부수는" 것만으론, 모델이 그 저분산 축을 인코딩할 *이유*가 여전히 없다. ⇒ SSL-only 개선은
*aug를 끄는 쪽*이 아니라 **절대 강도를 명시적 학습 target(pretext)으로 만드는 쪽**이어야 한다.

---

## 6. label 없이(SSL only) 개선하는 아이디어

원리: **절대 강도를 "invariance로 버릴 nuisance"에서 "예측해야 할 target"으로 전환**하고
(K1·K2 동시 공략), aug 수술은 그 *enabler*로만 둔다.

### 6-A. 절대 강도를 pretext target으로 (핵심 — K1·K2 동시 공략)

| 아이디어 | 메커니즘 | 비고 |
|---|---|---|
| **① MAE식 픽셀 복원 auxiliary** | 마스크 패치의 *원본 강도*를 복원하려면 backbone이 절대 강도를 보존할 수밖에 없음 | repo에 MAE 코드 있음(semi**_MAE_**). 가장 원리적·강력. iBOT(프로토타입)과 병행 |
| **② per-patch 평균강도 회귀 head** | 작은 head가 패치의 (aug 전) raw mean intensity 회귀 → DC 축에 capacity 강제 할당 | ①의 경량판. target은 **destructive photometric aug 적용 전** 강도여야 함 |
| **③ augmentation 파라미터 예측** | 두 view 사이 적용된 gamma/brightness를 예측 → invariance 대신 **equivariance** 학습 → 절대 광도 상태가 복원 가능해짐 | K1을 정면으로 뒤집음 |

> 공통 주의: target은 **파괴적 photometric aug가 적용되기 전**의 강도여야 의미가 있다. 안 그러면
> 손상된(augmented) 강도를 회귀하게 됨.

### 6-B. 절대 레벨을 고분산·구조적 신호로 "승격" (K2 우회)

- **④ 강도 기반 self-pseudo-label**: 패치 raw mean을 bin으로 묶어 보조 contrastive target.
  단 naive bin은 **반사 gradient까지 분리**(역설 악화)하므로, **region 단위로 풀링**해야 함 → ⑤.
- **⑤ 고전 CV prior를 free pseudo-sup으로 (가장 유망, 역설도 완화)**: SLIC superpixel /
  region-growing으로 over-segment → **region-mean intensity**를 신호로 사용. region-mean은
  내부 반사 gradient에 robust(평균으로 상쇄)하면서 인접 region 간 DC 차이는 보존 →
  **물질 분리 ↑ + 물질내 과분할 ↓**. 라벨 없는 weak-sup의 SSL 버전. 흐린 경계에서 superpixel
  품질이 관건.

### 6-C. 경계/대비 강화 (K3)

- **⑥ blur aug 약화 + 경계 보존**: disk_blur 확률·강도 ↓ (이미 v7에서 약화했지만 더).
- **⑦ self-supervised boundary 예측**: 평활화된 강도의 transition 위치를 예측하는 보조 task →
  경계 패치를 distinct하게. 흐린 경계엔 학습형 high-pass 결합 필요.

### 6-D. aug 수술 (필요조건, 단독으론 불충분 — §5)

- gamma/brightness 섭동 범위를 $\Delta\approx0.08$ **아래로** 클램프(K1 직접 완화).
- per-image norm → **fixed(dataset) norm**으로 절대 레벨 보존(단 ①~⑤와 *병행*해야 효과).

**우선순위 추천**: ⑤(region prior) ≳ ①(MAE 복원) > ③(aug 예측) > ②, 그리고 ⑥/⑦/6-D는 항상 병행.
①·⑤가 "신호를 target으로 만든다"는 점에서 기존 실패한 ablation들과 질적으로 다름.

---

## 7. 그래도 weak sup이 결국 필요한 이유 (Pareto ceiling)

위 SSL-only 아이디어들의 **공통 한계**: 전부 *절대 강도*는 살리지만 **material 자체는 모른다.**
그래서 강도를 살리면 물질 내부 반사 gradient도 같이 살아나 **과분할**되고, 억누르면 물질도 같이
병합된다:

$$
\text{merge materials} \;\longleftrightarrow\; \text{over-split within material}
$$

material prior(= "이 강도 변화는 같은 물질의 조명, 저 강도 변화는 다른 물질")가 없으면 이 **Pareto
front 위를 이동할 뿐 벗어나지 못한다.** ⑤(region prior)가 front를 가장 바깥으로 밀어주지만, 흐린
경계·애매 영역에서 결국 한계. **약한 label(weak sup)은 정확히 이 material prior를 주입**하여
front 자체를 깬다 — 그래서 SSL 개선은 *상한을 끌어올리는 보조*로, weak sup은 *천장을 뚫는 주력*으로
함께 가는 게 맞다.

---

## 8. 검증 제안

1. **K1 직접 측정**: gamma/poisson/blur 각 밝기 섭동 분포 vs $\Delta=0.082$ overlap.
2. **K2 정량**: 층별 A/B Fisher ratio $\dfrac{(\bar f_A-\bar f_B)^2}{\sigma_A^2+\sigma_B^2}$ — 깊을수록 하락?
3. **축 정렬(역설 핵심)**: 반사 gradient PCA 방향 $v_{\text{refl}}$ vs 물질 분리 LDA 방향
   $v_{\text{mat}}$의 $\cos\angle$. 직교면 push-only 안전, 평행이면 물질 push가 반사 과분할 악화.
4. **반사가 DC만 바꾸나 텍스처도 바꾸나** 정량 — 텍스처 차이 있으면 그걸 키로 유도 가능.

---

## 9. 정리 표

| | 메커니즘 | 강도 | 기존 관찰 연결 |
|---|---|---|---|
| **K1** | aug orbit이 클래스 gap 삼킴 ($\Delta<$ aug 섭동) | 강함(확실) | CLAHE off marginal |
| **K2** | LN null space + 저분산 → 압축에서 폐기 | 강함(확실) | 깊은 층일수록 sim[1,2] 악화 |
| **K3** | 흐린 경계 = edge 부재 + iBOT class-agnostic | 중간(메커니즘적) | info-aware masking harmful |
| 역설 | 절대 레벨 버리고 구조적 변화 살림 | 강함(관측+설명) | PCA 물질내 그라데이션 |
| 상한 | material prior 부재 → 병합↔과분할 Pareto | — | pure SSL 한계 결론 |

**한 줄 결론**: 범인은 "intensity invariance 일반"이 아니라 **"절대 레벨 ↔ 물질 ambiguity +
material prior 부재"**. SSL만으론 절대 강도를 *target화*(①·⑤)해 상한을 끌어올릴 수 있으나
Pareto ceiling이 있고, 천장은 weak sup(material prior 주입)으로만 뚫린다.

(weak loss 설계: [HANDOFF_CONTEXT.md](../HANDOFF_CONTEXT.md) §4, 구현:
[losses.py](../dino_v3/dinov3/train/weaksup/losses.py).)
