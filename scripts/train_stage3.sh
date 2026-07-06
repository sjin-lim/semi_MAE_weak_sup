#!/usr/bin/env bash
# ============================================================
# Progressive High-Resolution Upscaling (weak-sup 유지, light)
#
# 한 해상도로 적응 학습 → 그 teacher 로 다음(더 큰) 해상도 적응 → 반복.
#   기본: 768 → 1024 → 2048  (각 라운드가 이전 라운드 teacher 에서 resume)
#
# 왜 이 방식:
#   - DINOv3 는 RoPE(pos_embed_rope_rescale_coords) 라 해상도를 바꿔가며
#     weight 를 이어받아 확장 학습 가능.
#   - 한 번에 2k 로 점프하면 불안정 → 점진 확대가 안정적.
#   - 안정성을 위해 multi-res 리스트 config 대신 stage2 의 scalar-crop 경로를
#     베이스로 쓰고 해상도만 라운드별 override (검증된 path).
#
# 각 라운드 산출 teacher: <output>/res<R>/eval/<iter>/teacher_checkpoint.pth
# (do_test 는 eval_period 마다만 저장. eval_period=라운드iters 로 두면 "맨 끝 1회"만
#  발화 → 경계 off-by-one 으로 0개 될 수 있어, N_EVALS 로 라운드당 여러 번 발화시켜
#  chain 을 보장. 다음 라운드는 mtime 최신 teacher_checkpoint.pth 를 자동 선택.)
#
# ────────────────────────────────────────────────────────────
# 사용법
# ────────────────────────────────────────────────────────────
#   필수:
#     STAGE2_TEACHER   1라운드 시작점 (stage2 long 산출 teacher_checkpoint.pth)
#     SHARDS_PATH      WebDataset shards 경로
#     LABELED_ROOT     labeled_root (images/ + masks/)
#
#   기본 실행 (768 → 1024 → 2048):
#     STAGE2_TEACHER=/out/stage2_long/eval/training_XXXX/teacher_checkpoint.pth \
#     SHARDS_PATH=/data/shards \
#     LABELED_ROOT=/data/labeled \
#     OUTPUT_ROOT=./out/upscale \
#     NPROC=4 \
#     bash scripts/train_stage3.sh
#
#   해상도/배치/epoch 커스터마이즈 (라운드 수는 세 배열 길이가 일치해야 함):
#     RES_STEPS="768 1024"  BATCH_STEPS="16 6"  EPOCH_STEPS="12 8" \
#     STAGE2_TEACHER=... SHARDS_PATH=... LABELED_ROOT=... bash scripts/train_stage3.sh
#
#   중간부터 재개 (예: 768 은 끝났고 1024 부터):
#     RES_STEPS="1024 2048"  BATCH_STEPS="6 1"  EPOCH_STEPS="8 4" \
#     STAGE2_TEACHER=/out/upscale/res768/eval/training_XXXX/teacher_checkpoint.pth \
#     ... bash scripts/train_stage3.sh
#
#   ⚠️ 해상도(RES)는 반드시 patch_size(16)의 배수. (718 → 768 사용)
#   ⚠️ 해상도↑ 메모리 폭증(attention O(N²)) → BATCH_STEPS 를 크게 낮춰야 함.
#      2048 은 batch 1 도 OOM 가능 → 필요 시 LOCAL_NUM↓, flash-attn 확인.
# ============================================================
set -euo pipefail

# ── 필수 ────────────────────────────────────────────────────
STAGE2_TEACHER="${STAGE2_TEACHER:?set STAGE2_TEACHER=/path/to/stage2/eval/.../teacher_checkpoint.pth}"
SHARDS_PATH="${SHARDS_PATH:-/path/to/webdataset/shards}"
LABELED_ROOT="${LABELED_ROOT:-/path/to/labeled_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./out/upscale}"
NPROC="${NPROC:-4}"

# ── 라운드별 스케줄 (공백 구분, 세 배열 길이 일치) ───────────
#   RES   : 해상도 (16 배수).  BATCH : per-gpu batch.  EPOCH : 라운드 epoch.
read -r -a RES_STEPS   <<< "${RES_STEPS:-768 1024 2048}"
read -r -a BATCH_STEPS <<< "${BATCH_STEPS:-16 6 1}"
read -r -a EPOCH_STEPS <<< "${EPOCH_STEPS:-12 8 4}"

# ── 공통 하이퍼 (override 가능) ─────────────────────────────
LOCAL_DIV="${LOCAL_DIV:-4}"          # local crop = global / LOCAL_DIV (16 배수로 내림)
LOCAL_NUM="${LOCAL_NUM:-6}"          # local crop 개수 (hires 는 메모리 위해 축소)
LR_PEAK="${LR_PEAK:-2.0e-5}"         # hires 적응은 낮은 LR
LABELED_RATIO="${LABELED_RATIO:-0.15}"
LAMBDA_W="${LAMBDA_W:-10.0}"         # light preservation (stage2 30 → 10)
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
EPOCH_LEN="${EPOCH_LEN:-320}"        # OFFICIAL_EPOCH_LENGTH (아래서 override 로 강제)
N_EVALS="${N_EVALS:-4}"             # 라운드당 eval(=teacher 저장) 횟수. 마지막 1회만 두면
                                    # 경계 off-by-one 으로 0개 될 수 있어 여러 번 발화시킴

# scalar-crop 검증 경로 (multi-res 리스트 config 아님)
CONFIG="${CONFIG:-dino_v3/dinov3/configs/train/weaksup/stage2_ssl_weaksup.yaml}"

# ── 검증 ────────────────────────────────────────────────────
if [ "${#RES_STEPS[@]}" -ne "${#BATCH_STEPS[@]}" ] || [ "${#RES_STEPS[@]}" -ne "${#EPOCH_STEPS[@]}" ]; then
    echo "ERROR: RES_STEPS(${#RES_STEPS[@]}) / BATCH_STEPS(${#BATCH_STEPS[@]}) / EPOCH_STEPS(${#EPOCH_STEPS[@]}) 길이 불일치"
    exit 1
fi

# 라운드 산출물에서 최신 teacher_checkpoint.pth 찾기 (mtime 최신)
find_teacher() {
    local dir="$1" lines
    mapfile -t lines < <(find "${dir}/eval" -name teacher_checkpoint.pth -printf '%T@ %p\n' 2>/dev/null | sort -rn)
    [ "${#lines[@]}" -eq 0 ] && return 1
    echo "${lines[0]#* }"    # "timestamp path" → path
}

# ── 진행 ────────────────────────────────────────────────────
RESUME_CKPT="${STAGE2_TEACHER}"
N="${#RES_STEPS[@]}"

for i in "${!RES_STEPS[@]}"; do
    R="${RES_STEPS[$i]}"
    B="${BATCH_STEPS[$i]}"
    E="${EPOCH_STEPS[$i]}"

    if (( R % 16 != 0 )); then echo "ERROR: RES ${R} 는 16 배수여야 함"; exit 1; fi

    LOCAL=$(( (R / LOCAL_DIV / 16) * 16 ));  (( LOCAL < 16 )) && LOCAL=16
    COSINE=$(( E - WARMUP_EPOCHS ));         (( COSINE < 1 )) && COSINE=1
    ROUND_ITERS=$(( E * EPOCH_LEN ))
    # eval 을 라운드당 N_EVALS 번 발화 (마지막 경계 miss 방지). max_iter 를 EPOCH_LEN
    # override 로 통제하니 EVAL_PERIOD 는 ROUND_ITERS 를 정확히 나눔.
    EVAL_PERIOD=$(( ROUND_ITERS / N_EVALS )); (( EVAL_PERIOD < 1 )) && EVAL_PERIOD=${ROUND_ITERS}
    OUT="${OUTPUT_ROOT}/res${R}"
    mkdir -p "${OUT}"

    echo "======================================================="
    echo "[upscale] round $((i+1))/${N}  res=${R}  batch=${B}  epochs=${E}  local=${LOCAL}x${LOCAL_NUM}"
    echo "[upscale] resume : ${RESUME_CKPT}"
    echo "[upscale] output : ${OUT}"
    echo "[upscale] iters=${ROUND_ITERS}  eval_period=${EVAL_PERIOD} (x${N_EVALS} teacher 저장)"
    echo "======================================================="

    torchrun --nproc_per_node="${NPROC}" \
        dino_v3/dinov3/train/train.py \
        --config-file "${CONFIG}" \
        --no-resume \
        "train.dataset_path=WebDataset:path=${SHARDS_PATH}" \
        "train.output_dir=${OUT}" \
        "student.resume_from_teacher_chkpt=${RESUME_CKPT}" \
        "weak_sup.labeled_root=${LABELED_ROOT}" \
        "weak_sup.labeled_ratio=${LABELED_RATIO}" \
        "weak_sup.lambda_W=${LAMBDA_W}" \
        "weak_sup.batch_size_labeled=${B}" \
        "crops.global_crops_size=${R}" \
        "crops.local_crops_size=${LOCAL}" \
        "crops.local_crops_number=${LOCAL_NUM}" \
        "train.batch_size_per_gpu=${B}" \
        "optim.epochs=${E}" \
        "train.OFFICIAL_EPOCH_LENGTH=${EPOCH_LEN}" \
        "schedules.lr.peak=${LR_PEAK}" \
        "schedules.lr.warmup_epochs=${WARMUP_EPOCHS}" \
        "schedules.lr.cosine_epochs=${COSINE}" \
        "evaluation.eval_period_iterations=${EVAL_PERIOD}" \
        "$@"

    NEXT_CKPT="$(find_teacher "${OUT}" || true)"
    if [ -z "${NEXT_CKPT}" ]; then
        echo "ERROR: ${OUT}/eval 에서 teacher_checkpoint.pth 없음."
        echo "       eval 이 안 돌았을 수 있음 (eval_period_iterations=${EVAL_PERIOD} vs iters=${ROUND_ITERS})."
        echo "       eval 폴더 내용:"
        ls -la "${OUT}/eval" 2>/dev/null || echo "         (eval 폴더 자체가 없음)"
        echo "       sharded 로 저장됐다면 sharded_teacher_checkpoint/ 디렉토리를 RESUME 로 직접 지정 가능."
        exit 1
    fi
    echo "[upscale] round $((i+1)) 완료 → teacher: ${NEXT_CKPT}"
    RESUME_CKPT="${NEXT_CKPT}"
done

echo "======================================================="
echo "[upscale] ALL DONE (${N} rounds). final teacher:"
echo "  ${RESUME_CKPT}"
echo "======================================================="
