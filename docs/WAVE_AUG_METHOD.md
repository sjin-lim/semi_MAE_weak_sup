# Wave Augmentation 생성 방식 (수식 설명)

> EM 반사(ripple) artifact 를 모사하는 augmentation 의 생성 원리·수식.
> 목적: student view 에만 적용해 모델이 **wave 에 invariant** 해지도록(= "이 빛 변화는 noise") 학습.
> 배경/근거: [ANALYSIS_grayscale_separation.md](ANALYSIS_grayscale_separation.md) §6.
> 도구/검증: [../notebooks/wave_artifact_analysis.ipynb](../notebooks/wave_artifact_analysis.ipynb).

---

## 1. wave 모델

관측 이미지를 "물질 구조 × 곱셈 ripple" 로 본다:

$$
I_{\text{obs}}(p) \;=\; I_{\text{material}}(p)\,\big(1 + a\,w(p)\big)
$$

- $w(p)$ : **zero-mean ripple field**, 특정 공간주파수 band $[\,k_{lo}, k_{hi}\,]$ 에 집중
- $a$ : 진폭 (실측 매칭, §4)

곱셈으로 두는 이유: 반사는 밝은 영역을 더 크게 흔드는 *gain* 성격(조명/반사율)이기 때문.
가산($I + a w$)도 가능하나 곱셈이 물리에 더 가깝다.

---

## 2. ripple field $w$ 생성 — band-pass noise

단일 sinusoid 대신 **band 전체를 덮는 band-pass 필터링 노이즈**로 만든다:

1. white noise $n(p)\sim\mathcal N(0,1)$
2. $\hat n = \mathcal F\{n\}$ (2D FFT)
3. **annulus mask** 로 band 만 통과 (방향성 있으면 각도 wedge 추가):
$$
M(\mathbf k) = \mathbb 1\big[k_{lo}\le \lVert\mathbf k\rVert \le k_{hi}\big]\;\cdot\;\mathbb 1\big[\,|\angle\mathbf k - \theta|\le \Delta\theta\,\big]
$$
4. 역변환 후 실수부: $\tilde w = \mathrm{Re}\,\mathcal F^{-1}\{\hat n \odot M\}$
5. 정규화: $\displaystyle w = \mathrm{clip}\!\Big(\tfrac{\tilde w - \overline{\tilde w}}{\mathrm{std}(\tilde w)},\,-3,\,3\Big)\big/3$
   → $w$ 는 **zero-mean**, 대략 $[-1,1]$, $\mathrm{std}(w)\approx 1/3$.

**왜 band-pass noise 인가**
- 실제 간섭무늬의 **spectral signature(주파수 band)** 를 직접 모사 → 그 band 전체에 invariance.
- 매 호출마다 **다른 비반복 패턴**(phase/방향 자동 랜덤) → 모델이 특정 패턴을 외워 회피 불가.
- 단일 cos 는 한 주파수·위상만 → 일반화 약함.

---

## 3. 적용 — mean-preserving (다크닝 아님)

곱셈(mult) 모드:

$$
I' = I\,(1 + a\,w)
$$

**공간 평균 보존** (= 전체적으로 밝아지거나 어두워지지 않음):

$$
\overline{I'} = \overline{I} + a\,\overline{I\odot w}
= \overline{I} + a\big(\underbrace{\overline I\,\overline w}_{=0}\; + \;\mathrm{cov}(I,w)\big)
\;\approx\; \overline{I}
$$

$w$ 가 zero-mean 이고 구조와 (거의) 무상관이라 평균 이동 $\approx 0$. (실측: $\Delta\overline I \approx 0.0000$.)
factor $1+a w \in [1-a,\,1+a]$ 로 **밝게/어둡게 대칭** — 즉 ripple 이지 darkening 이 아니다.

log(homomorphic) 모드:

$$
I' = \exp\big(\log I + a\,w\big) = I\cdot e^{a w}
$$

Jensen 부등식으로 $\mathbb E[e^{aw}] = e^{a^2\sigma_w^2/2} > 1$ → **아주 약간 밝아짐**(다크닝의 반대).
실측 $\Delta\overline I \approx +0.001$ 수준. saturation 비대칭이 걱정되면 log 모드 사용.

> **"어두워 보이는" 착시 주의**: 밝은 EM 이미지에서 (a) 어두운 골이 밝은 마루보다 눈에 더 띄고
> (Weber–Fechner), (b) 디스플레이에서 `clip(·,0,1)` 하면 밝은 마루(×$(1{+}a)$)가 1에 붙어 증가분이
> 안 보임 → 어둡게 *보임*. 수치는 대칭이고 **float 유지 시 clip 자체가 없어 정보 손실 없음.**

---

## 4. 진폭 $a$ 를 실제 wave 에 맞추기

목표는 **실제 ripple 진폭 범위 전체에 invariant** 하게 만드는 것. 약하게만 주면 약한 wave 에만
무뎌진다. 실측 band 에너지로 $a$ 를 정한다.

실제 이미지의 band-pass 성분 $r = \mathcal F^{-1}\{\hat I\odot M\}$ 의 상대 std:

$$
\sigma^{\text{rel}}_{\text{real}} = \frac{\mathrm{std}(r)}{\overline I}
\quad(\text{with/without 그룹 있으면 분산 차로 baseline 제거})
$$

적용 섭동 $a w$ 의 std 를 여기 맞춤 → $a\,\mathrm{std}(w) = \sigma^{\text{rel}}_{\text{real}}$:

$$
\boxed{\,a \;=\; \dfrac{\sigma^{\text{rel}}_{\text{real}}}{\mathrm{std}(w)}\,}\qquad(\mathrm{std}(w)\approx 1/3)
$$

학습 시 `amp_jitter=True` 로 $a\sim\mathcal U[0, a_{\max}]$ → **no-wave 부터 실제 강도까지** 커버
(있는/없는 이미지 혼합도 자연 처리).

---

## 5. 학습 파이프라인 적용 규칙

- **student global view 에만** 적용, teacher 미적용 → teacher(clean anchor) ↔ student(waved) 의
  **DINO/iBOT consistency 가 "wave=noise" 를 자동 학습** (별도 loss/head 불필요).
- 적용 확률 $p$ (예: 0.5) + `amp_jitter` → 분포에 no-wave 포함.
- **전역** wave 기본, 가끔 **가우시안 envelope** 로 국소(물질 내부) wave 도 모사:
$$
w_{\text{local}}(p) = w(p)\cdot \exp\!\Big(-\tfrac{\lVert p - c\rVert^2}{2\sigma^2}\Big)
$$
- `gamma` 전역 aug 는 계속 off (물질 DC 채널 보존), geometric/masking 은 유지(SSL 핵심).

---

## 6. 왜 이게 도움이 되는가 (요약)

- 분석(M1): **반사 variance $\sigma_{\text{reflection}}$ 가 표현 capacity 를 독식해 물질 DC 를 masking.**
- wave-aug invariance = $\sigma_{\text{reflection}}$ 를 **표현에서 제거** → SNR 분모↓ → 묻혀있던 물질
  신호가 표면으로.
- weak sup(per-patch push)의 **enabler**: per-patch feature 의 wave 잡음을 걷어 push 를 물질 신호
  위에서 동작하게 함.
- 분업: **wave-aug = within-region 반사 제거**, **weak push = between-region 물질 분리**
  → 단일-knob Pareto confound 를 부분 회피 ([ANALYSIS](ANALYSIS_grayscale_separation.md) §7.2).
