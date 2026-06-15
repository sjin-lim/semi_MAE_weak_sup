# Wave Augmentation — 방법 · 수식 · 준비 워크플로

> EM 반사(ripple) artifact 를 모사하는 augmentation 의 생성 원리·수식과,
> "wave 특성을 어떻게 준비하는가"의 전체 흐름.
> 목적: student view 에만 적용해 모델이 **wave 에 invariant** 해지도록(= "이 빛 변화는 noise") 학습.
> 배경/근거: [ANALYSIS_grayscale_separation.md](ANALYSIS_grayscale_separation.md) §6.
> 측정/검증 도구: [../notebooks/wave_artifact_analysis.ipynb](../notebooks/wave_artifact_analysis.ipynb).
> 구현: [../dino_v3/dinov3/data/augmentations.py](../dino_v3/dinov3/data/augmentations.py) (`make_wave_field`, `WaveModulation`).

---

## 0. 큰 그림 — "특성 준비"는 오프라인 1회, "물결 생성"은 온라인 매번

wave-aug 는 **두 층**으로 나뉜다. 헷갈리지 않는 게 핵심:

| 층 | 무엇 | 언제 | 어디 |
|---|---|---|---|
| **특성(characteristic)** | band `[k_lo,k_hi]`, 진폭 `amp`, 방향 `direction/dir_width` — **스칼라 4개** | **오프라인 1회** 실측 | `crops.wave_aug` config |
| **물결 필드(field)** | 그 특성 안의 *구체적* ripple 한 장 (모양·위상) | **온라인 매 이미지** 난수 합성 | `make_wave_field()` 런타임 |

- 실제 반사 패턴을 **샘플링/뱅킹하지 않는다.** 측정하는 건 *스펙트럼 envelope(band+amp)* 뿐.
- 그 envelope 안에서 **매번 다른 난수 ripple** 을 즉석 생성 → 무한 다양성, 미리 만들 필요 없음.
- 즉 "준비" = **§4 의 실측값을 config 4개 스칼라로 박아두는 것**이 전부.

---

## 1. wave 모델

관측 이미지를 "물질 구조 × 곱셈 ripple" 로 본다:

$$
I_{\text{obs}}(p) \;=\; I_{\text{material}}(p)\,\big(1 + a\,w(p)\big)
$$

- $w(p)$ : **zero-mean, unit-std ripple field**, 공간주파수 band $[\,k_{lo}, k_{hi}\,]$ 에 집중
- $a$ : 상대 진폭 (실측 매칭, §4). $w$ 가 unit-std 라 $\mathrm{std}(a\,w)=a$ = **상대 ripple std** 그 자체

곱셈으로 두는 이유: 반사는 밝은 영역을 더 크게 흔드는 *gain* 성격(조명/반사율)이기 때문.
가산($I + a w$)도 가능하나 곱셈이 물리에 더 가깝다.

---

## 2. ripple field $w$ 생성 — band-pass noise (unit-std)

단일 sinusoid 대신 **band 전체를 덮는 band-pass 필터링 노이즈**로 만든다:

1. white noise $n(p)\sim\mathcal N(0,1)$
2. $\hat n = \mathcal F\{n\}$ (2D FFT)
3. **annulus mask** 로 band 만 통과 (방향성 있으면 각도 wedge 추가):
$$
M(\mathbf k) = \mathbb 1\big[k_{lo}\le \lVert\mathbf k\rVert \le k_{hi}\big]\;\cdot\;\mathbb 1\big[\,|\angle\mathbf k - \theta|\le \Delta\theta\,\big]
$$
4. 역변환 후 실수부: $\tilde w = \mathrm{Re}\,\mathcal F^{-1}\{\hat n \odot M\}$
5. **unit-std 정규화** + 안전 clip:
$$
w = \mathrm{clip}\!\Big(\tfrac{\tilde w - \overline{\tilde w}}{\mathrm{std}(\tilde w)},\,-4,\,4\Big)
\quad\Rightarrow\quad \overline w = 0,\;\; \mathrm{std}(w)\approx 1
$$
   → 따라서 적용 시 $\mathrm{std}(a w)=a$. **`amp` 가 곧 상대 ripple std** 가 되어 §4 매칭이 1:1.

> ⚙️ **속도 트릭 (lowres + upsample, ~10×).** band 가 저주파($k_{hi}$ 작음)라, 작은 격자 $h\times h$
> ($h=\mathrm{clip}(\lceil S\,k_{hi}/0.4\rceil,\,32,\,S)$)에서 생성 후 $S\times S$ 로 bilinear 업샘플.
> 저주파 정보는 손실 없음. crop 1장당 full-FFT 대신 작은 FFT → 학습 병목 회피.

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

$w$ 가 zero-mean 이고 구조와 (거의) 무상관이라 평균 이동 $\approx 0$. factor $1+a w$ 로 **밝게/어둡게 대칭** — ripple 이지 darkening 이 아니다.

log(homomorphic) 모드:

$$
I' = \exp\big(\log I + a\,w\big) = I\cdot e^{a w}
$$

Jensen 부등식으로 $\mathbb E[e^{aw}] = e^{a^2\sigma_w^2/2} > 1$ → **아주 약간 밝아짐**(다크닝의 반대).
saturation 비대칭이 걱정되면 log 모드 사용.

> **"어두워 보이는" 착시 주의**: 밝은 EM 이미지에서 (a) 어두운 골이 밝은 마루보다 눈에 더 띄고
> (Weber–Fechner), (b) 디스플레이에서 `clip(·,0,255)` 하면 밝은 마루(×$(1{+}a)$)가 상한에 붙어
> 증가분이 안 보임 → 어둡게 *보임*. 수치는 대칭이고 정보 손실은 clip 된 극단부뿐이다.

---

## 4. 진폭 $a$ 를 실제 wave 에 맞추기 (← 핵심 "준비")

목표는 **실제 ripple 진폭 범위에 invariant** 하게 만드는 것. 약하게만 주면 약한 wave 에만
무뎌진다. 실측 band 에너지로 $a$ 를 정한다.

실제 이미지의 band-pass 성분 $r = \mathcal F^{-1}\{\hat I\odot M\}$ 의 상대 std:

$$
\sigma^{\text{rel}}_{\text{real}} = \frac{\mathrm{std}(r)}{\overline I}
$$

$w$ 가 **unit-std** 이므로 적용 섭동 std 매칭이 곧:

$$
\boxed{\,a \;=\; \dfrac{\sigma^{\text{rel}}_{\text{real}}}{\mathrm{std}(w)} \;=\; \sigma^{\text{rel}}_{\text{real}}\,}\qquad(\mathrm{std}(w)\approx 1)
$$

즉 **`amp` 에 실측 `real_rel_std` 를 그대로 넣으면 "실제 세기"** 가 된다.

> ⚠️ **보수적으로 깐다.** 실측 $\sigma^{\text{rel}}_{\text{real}}\approx 0.30$ 은 학습용으론 과함
> (물질 DC=색차 채널을 침범할 위험, [ANALYSIS](ANALYSIS_grayscale_separation.md) §6). 그래서
> **실측의 30~50% (현재 `amp=0.12`)** 로 "얼룩덜룩" 수준만. 학습 시 `amp_jitter=True` 로
> $a\sim\mathcal U[0, a_{\max}]$ → **no-wave 부터 채택 강도까지** 커버(있는/없는 이미지 혼합 자연 처리).

**측정 절차 (notebook):** [wave_artifact_analysis.ipynb](../notebooks/wave_artifact_analysis.ipynb)
§4 = radial power spectrum 으로 `k_lo/k_hi`(+방향), §4b = `real_rel_std` → `amp` 후보.
새 데이터셋이면 이 두 셀만 다시 돌려 config 갱신.

---

## 5. 왜 옅은 경계(major)는 안 망가지나 — 주파수 분리

- **물질 경계 = step edge = 광대역(고주파 포함).** ripple band `[0.001, 0.0031]` 은 **저주파 좁은 밴드**
  → step edge 의 고주파 성분은 거의 안 건드림 → 경계는 살아남음.
- **유일한 회색지대**: 두 큰 영역의 near-DC 차이(147 vs 168 이 큰 면적일 때 $\sim k_{lo}$ 근처).
  → 가드레일: `k_lo` 너무 낮추지 말 것 + `amp` modest. (band 은 DC 미포함, $k_{lo}>0$.)

---

## 6. 학습 파이프라인 적용 규칙

- **student global view 에만** 적용, teacher 미적용 → teacher(clean anchor, `teacher_no_color_jitter`)
  ↔ student(waved) 의 **DINO/iBOT consistency 가 "wave=noise" 를 자동 학습** (별도 loss/head 불필요).
  - labeled 경로(`LabeledImageAugmentation`)는 clean anchor 가 없어, 두 global crop 에 **독립 wave
    realization** 을 줘 cross-view 로 동일 효과. mask 는 곱셈·mean보존·기하불변이라 정합 영향 없음.
- 적용 확률 $p$ (예: 0.5) + `amp_jitter` → 분포에 no-wave 포함.
- **전역** wave 기본, `spatial_envelope_p` 확률로 **가우시안 envelope** 국소(물질 내부) wave 도 모사:
$$
w_{\text{local}}(p) = w(p)\cdot \exp\!\Big(-\tfrac{\lVert p - c\rVert^2}{2\sigma^2}\Big)
$$
- `gamma`/`clahe` 전역 aug 는 계속 off (물질 DC 채널 보존), geometric/masking 은 유지(SSL 핵심).

---

## 7. config 레퍼런스 (`crops.wave_aug`)

```yaml
crops:
  wave_aug:
    enabled: true        # off 면 wave 미적용 (build_wave_modulation → None)
    k_lo: 0.001          # ripple band 하한 (cycles/pixel), 실측 §4
    k_hi: 0.0031         # 상한 (저주파라 step 경계[고주파] 보존)
    amp: 0.12            # 상대 ripple std = real_rel_std(0.30) 의 ~40% (보수적)
    direction: null      # 등방성. 방향성 반사면 deg 값
    dir_width: 30.0      # 방향 wedge ± 폭 (deg)
    p: 0.5               # 적용 확률 (있는/없는 이미지 자연 혼합)
    amp_jitter: true     # amp 를 [0, amp] 랜덤
    mode: mult           # mean 보존 곱셈 (또는 log: homomorphic)
    spatial_envelope_p: 0.3   # 이 확률로 국소(물질 내부) ripple
```

- 코드 진입점: `build_wave_modulation(cfg)` → `WaveModulation`. `enabled:false` 또는 키 누락 시 자동 off.
- 적용 위치: `FusedEMIntensity` **뒤**, `normalize` **앞** (uint8 PIL 에 곱셈).

---

## 8. 전체 전략에서의 위치 (분업)

| 문제 | 레버 | 성격 |
|---|---|---|
| **옅은 경계 (major)** | 해상도(hires) + **hinge anti-merge** weak loss (margin 0.85) | 합침 방지 (정보 보존) |
| **물질 내 밝기/반사 (minor)** | **wave-aug** (student-only consistency) | nuisance 무시 학습 (비파괴) |
| ~~pull loss~~ | — | **거부**: class 내 변이 영구 삭제 = 정보 파괴 + per-task 충돌 |

→ wave-aug 는 **within-region 반사 제거**, hinge weak loss 는 **between-region 합침 방지**.
둘 다 *정보를 지우지 않아* frozen-backbone + per-task-decoder 배포 철학과 정합.
상세: [ANALYSIS_grayscale_separation.md](ANALYSIS_grayscale_separation.md) §6–§7.
