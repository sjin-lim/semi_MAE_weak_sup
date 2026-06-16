# DINO feature 분리도 / few-shot 진단

학습된 DINOv3(teacher) backbone 의 frozen feature 가 **폴더로 구분한 클래스**를
얼마나 잘 분리하는지 빠르게 검사하는 도구. 분산(distributed) 없이 **단일 GPU** 로 동작.

관련 파일:
- [fewshot_separability.py](fewshot_separability.py) — 진단 본체
- [em_aug.py](em_aug.py) — EM classification augmentation
- [../../../scripts/eval_separability.sh](../../../scripts/eval_separability.sh) — 실행 래퍼

---

## 1. 데이터 준비 (ImageFolder)

클래스별로 폴더를 나눠 이미지를 넣는다.

```
DATA_ROOT/
  classA/  img001.png  img002.png  ...
  classB/  ...
  classC/  ...
```

- 폴더 이름 = 클래스 이름. 2개 이상 필요.
- grayscale/RGB 무관 (내부에서 3채널 처리).

---

## 2. 빠른 실행

```bash
# (서버에서, 로컬엔 torch 없음)
cd dino_v3   # repo 루트가 아니라 dino_v3 안에서 실행

DATA_ROOT=/path/to/class_folders \
WEIGHTS=/path/to/teacher_checkpoint.pth \
bash ../scripts/eval_separability.sh
```

`WEIGHTS` 는 consolidated `.pth` 또는 DCP 체크포인트 **디렉토리** 모두 가능.

### 환경변수 (래퍼)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DATA_ROOT` | (필수) | ImageFolder 루트 |
| `WEIGHTS` | (필수) | teacher 체크포인트 |
| `CONFIG` | stage2_ssl_weaksup.yaml | 학습에 쓴 config (arch 일치용) |
| `OUTPUT_DIR` | ./out/separability | 결과 저장 |
| `IMAGE_SIZE` | 448 | 학습 global crop 과 동일 권장 |
| `FEATURE` | concat | `cls` / `patchmean` / `concat` |
| `SHOTS` | (빈값) | 비우면 50/50 stratified, 숫자면 K-shot |
| `N_TRIES` | 5 | 에피소드 반복 (평균±std) |
| `TRAIN_AUG` | 0 | 1이면 support feature augmentation |
| `AUG_VIEWS` | 4 | 이미지당 augmentation view 수 |
| `CROP_SCALE` | 0.8 | RandomResizedCrop scale 하한 |

### 직접 호출 (python)

```bash
python dinov3/eval/fewshot_separability.py \
    --config-file dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml \
    --pretrained-weights /path/to/teacher_checkpoint.pth \
    --data-root /path/to/class_folders \
    --image-size 448 --feature concat \
    --shots 20 --n-tries 5 \
    --train-aug --aug-views 4 \
    --output-dir ./out/separability
```

---

## 3. augmentation (선택)

`--train-aug` 켜면 **support(train) 이미지에만** augmentation view 를 `--aug-views`개
추가로 뽑아 분류기 학습 데이터를 늘린다. query/test 는 항상 clean.

포함 항목 ([em_aug.py](em_aug.py)): 밝기(감마)/90도 회전/H·V flip/가우시안 노이즈/
0.8배 크롭/검은 점(작은 dark dots).

> ⚠️ **중요**: 학습이 `instance_norm`(per-image 표준화)이라 eval 도 PerImageNormalize 로
> 끝낸다. 이때 **선형 밝기/대비는 정규화로 완전히 상쇄**되어 효과가 사라진다.
> → 밝기는 선형 scale 이 아니라 **감마(비선형)** 로 넣었다.
> 노이즈/검은점/기하 변환은 표준화 후에도 효과가 남는다.

`--train-aug` 유무로 두 번 돌려 정확도 차이를 보면 augmentation 효과를 바로 확인.

---

## 4. 결과 해석

출력: 콘솔 요약 + `OUTPUT_DIR/separability_report.json` + `embedding.png`(t-SNE/PCA).

| 지표 | 의미 |
|---|---|
| `silhouette(cosine)` | 높을수록 군집 분리 좋음 (−1~1) |
| `inter-class cosine mean` | 클래스 prototype 간 평균 cosine. **낮을수록** 분리 잘됨 |
| `ncm` | Nearest Class Mean (학습 0) — feature 자체 분리도 sanity |
| `knn` | kNN (학습 0) |
| `logreg` | Logistic Regression — 10~50 shot 본 성능 |
| per-class / confusion | 어느 클래스끼리 섞이는지 |

해석 가이드:
- **inter-class cosine 낮고 NCM/logreg 높음** → DINO feature 가 이미 분리. 바로 few-shot 본선.
- **NCM 낮은데 logreg 높음** → 선형으론 갈리나 군집 중심이 안 떨어짐. weak-sup 추가 학습이 도움 신호.
- **둘 다 낮음** → feature 가 클래스 신호를 안 담음. 폴더 클래스 정의 점검 또는 Stage2/3 weak-sup 학습 필요.

---

## 5. 주의

- **정규화 일치**: 학습이 `crops.instance_norm: true` 이므로 eval 도 PerImageNormalize.
  도구에 내장되어 있으니 신경 안 써도 됨 (직접 다른 eval 짤 때 주의).
- **config-file**: 학습에 쓴 것과 동일하게. arch(vit_large, patch16, n_storage_tokens=4)가
  config 로 결정됨.
- teacher 로드: `build_model_for_eval(only_teacher=True)` — student 아님.
