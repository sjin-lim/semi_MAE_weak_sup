#!/usr/bin/env python3
"""
Pretrained 체크포인트 → DINOv3 SSLMetaArch 초기화 weight 변환 스크립트

지원하는 입력 형식:
  1. MAE 체크포인트       : {"model": {patch_embed.*, blocks.*, norm.*}}
  2. DINOv3 체크포인트   : {"teacher": {"backbone.*", "dino_head.*", ...}}
                           (공식 pretrained or 이전 학습 결과물)
  3. 일반 ViT state_dict  : {patch_embed.*, blocks.*, norm.*}  (bare dict)

변환 동작:
  - 입력 형식 자동 감지
  - backbone 가중치를 DINOv3 ViT 구조로 이전 (strict=False)
  - DINOv3-only 파라미터(storage_tokens, LayerScale, RoPE, bias_mask)는
    올바른 초기값 유지
  - dino_head / ibot_head 는 config 크기에 맞게 랜덤 초기화
    (MAE → 새로 만든 head / DINOv3 공식 → head 크기가 달라도 재구성)
  - 출력: {"teacher": {"backbone.*", "dino_head.*", "ibot_head.*"}}
    → config 의 student.resume_from_teacher_chkpt 에 경로를 직접 지정 가능

사용 예:
  # MAE 체크포인트
  python tools/convert_to_dino_init.py \\
      --ckpt /path/to/mae_checkpoint.pth \\
      --output /path/to/dino_init.pth

  # DINOv3 공식 weight (head 크기가 달라도 자동 처리)
  python tools/convert_to_dino_init.py \\
      --ckpt /path/to/dinov3_vitl.pth \\
      --output /path/to/dino_init.pth

  # 아키텍처를 config 와 맞출 때 (기본값은 sem_mae_em_pretrain.yaml 기준)
  python tools/convert_to_dino_init.py \\
      --ckpt /path/to/checkpoint.pth \\
      --output /path/to/dino_init.pth \\
      --arch vit_large --img_size 448 \\
      --dino_prototypes 65536 --ibot_prototypes 32768

학습 config:
  student:
    resume_from_teacher_chkpt: '/path/to/dino_init.pth'
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).parent.parent
DINO_ROOT  = REPO_ROOT / "dino_v3"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DINO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 입력 형식 감지 및 backbone state_dict 추출
# ─────────────────────────────────────────────────────────────────────────────

# MAE 체크포인트에서 제외할 키
_MAE_SKIP_EXACT    = {"pos_embed", "mask_token"}
_MAE_SKIP_PREFIXES = ("decoder_", "head.")


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _detect_and_extract(raw: dict) -> tuple[dict, str]:
    """
    체크포인트에서 backbone raw state_dict 추출 + 감지된 형식 반환.
    반환: (backbone_sd, fmt)
      fmt: "mae" | "dinov3" | "bare"
    """
    # ── DINOv3 형식: ckpt["teacher"]["backbone.*"] ──────────────────────────
    if "teacher" in raw:
        teacher_sd = raw["teacher"]
        if any(k.startswith("backbone.") for k in teacher_sd):
            backbone_sd = _strip_prefix(teacher_sd, "backbone.")
            return backbone_sd, "dinov3"

    # ── MAE 형식: ckpt["model"] 또는 ckpt["state_dict"] ─────────────────────
    for key in ("model", "state_dict"):
        if key in raw:
            sd = raw[key]
            # prefix 자동 제거 (encoder.*, backbone.*, module.*)
            first = next(iter(sd))
            for prefix in ("encoder.", "backbone.", "module."):
                if first.startswith(prefix):
                    sd = _strip_prefix(sd, prefix)
                    break
            return sd, "mae"

    # ── bare state_dict (직접 저장된 경우) ───────────────────────────────────
    first = next(iter(raw))
    for prefix in ("encoder.", "backbone.", "module."):
        if first.startswith(prefix):
            return _strip_prefix(raw, prefix), "mae"
    return raw, "bare"


def load_backbone_weights(path: str) -> tuple[dict, str]:
    logger.info(f"[1/5] 체크포인트 로드: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd, fmt = _detect_and_extract(raw)
    logger.info(f"      감지된 형식: {fmt.upper()}")

    if fmt in ("mae", "bare"):
        # decoder, head, pos_embed, mask_token 제거
        kept, removed = {}, []
        for k, v in sd.items():
            if k in _MAE_SKIP_EXACT or any(k.startswith(p) for p in _MAE_SKIP_PREFIXES):
                removed.append(k)
            else:
                kept[k] = v
        if removed:
            logger.info(f"      MAE-only 키 제거 ({len(removed)}): {removed}")
        sd = kept

    logger.info(f"      backbone 키: {len(sd)}개")
    return sd, fmt


# ─────────────────────────────────────────────────────────────────────────────
# DINOv3 backbone 구성
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone(args) -> nn.Module:
    from dinov3.models import vision_transformer as vits

    logger.info(f"[2/5] DINOv3 backbone 구성")
    logger.info(f"      arch={args.arch}  patch={args.patch_size}  img={args.img_size}")
    logger.info(f"      n_storage_tokens={args.n_storage_tokens}  "
                f"mask_k_bias={args.mask_k_bias}  norm_layer={args.norm_layer}")

    model = getattr(vits, args.arch)(
        patch_size=args.patch_size,
        img_size=args.img_size,
        n_storage_tokens=args.n_storage_tokens,
        mask_k_bias=args.mask_k_bias,
        norm_layer=args.norm_layer,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        ffn_layer="mlp",
        layerscale_init=1e-5,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="bf16",
    )
    # bias_mask 패턴, storage_tokens, rope_embed.periods 등 올바른 초기화
    model.init_weights()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 가중치 이전
# ─────────────────────────────────────────────────────────────────────────────

def transfer_to_backbone(backbone: nn.Module, src_sd: dict):
    logger.info("[3/5] 가중치 이전 (strict=False)")
    result = backbone.load_state_dict(src_sd, strict=False)

    n_src        = len(src_sd)
    n_unexpected = len(result.unexpected_keys)
    n_transferred = n_src - n_unexpected
    n_missing     = len(result.missing_keys)

    logger.info(f"\n      {'─'*56}")
    logger.info(f"      이전 결과:")
    logger.info(f"        전이 성공  : {n_transferred} 개")
    logger.info(f"        Missing    : {n_missing} 개  (DINOv3-only, 초기값 유지)")

    # 항목별로 그룹핑해서 출력
    groups: dict[str, list] = {}
    for k in sorted(result.missing_keys):
        g = k.split(".")[0]
        groups.setdefault(g, []).append(k)
    for g, keys in groups.items():
        logger.info(f"          [{g}]  {len(keys)}개   예: {keys[0]}")

    if result.unexpected_keys:
        logger.info(f"        Unexpected : {n_unexpected} 개  (소스-only, 무시)")
        for k in sorted(result.unexpected_keys):
            logger.info(f"          {k}")
    logger.info(f"      {'─'*56}\n")


# ─────────────────────────────────────────────────────────────────────────────
# DINOHead 구성 (랜덤 초기화)
# ─────────────────────────────────────────────────────────────────────────────

def build_heads(args, embed_dim: int) -> tuple[nn.Module, nn.Module]:
    from dinov3.layers.dino_head import DINOHead

    logger.info(f"[4/5] DINO / iBOT head 랜덤 초기화")
    logger.info(f"      DINO: {embed_dim}→{args.dino_prototypes}  "
                f"hidden={args.dino_hidden_dim}  bottleneck={args.head_bottleneck_dim}")
    logger.info(f"      iBOT: {embed_dim}→{args.ibot_prototypes}  "
                f"hidden={args.ibot_hidden_dim}  bottleneck={args.head_bottleneck_dim}")

    def make_head(out_dim, hidden_dim):
        h = DINOHead(in_dim=embed_dim, out_dim=out_dim,
                     hidden_dim=hidden_dim, bottleneck_dim=args.head_bottleneck_dim,
                     nlayers=3)
        h.init_weights()
        return h

    return make_head(args.dino_prototypes, args.dino_hidden_dim), \
           make_head(args.ibot_prototypes, args.ibot_hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────────────────────

def save(backbone, dino_head, ibot_head, output_path: str):
    logger.info("[5/5] DINOv3 체크포인트 형식으로 저장")

    # init_fsdp_model_from_checkpoint 가 요구하는 구조:
    #   ckpt["teacher"] → self.student(ModuleDict) 에 strict=True 로 로드
    #   self.student 키: backbone.* | dino_head.* | ibot_head.*
    teacher_sd = {}
    for k, v in backbone.state_dict().items():
        teacher_sd[f"backbone.{k}"] = v
    for k, v in dino_head.state_dict().items():
        teacher_sd[f"dino_head.{k}"] = v
    for k, v in ibot_head.state_dict().items():
        teacher_sd[f"ibot_head.{k}"] = v

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": teacher_sd}, out)

    size_mb   = out.stat().st_size / 1e6
    n_total   = sum(v.numel() for v in teacher_sd.values()
                    if isinstance(v, torch.Tensor) and v.dtype.is_floating_point)
    n_backbone = sum(v.numel() for k, v in teacher_sd.items()
                     if k.startswith("backbone.") and isinstance(v, torch.Tensor)
                     and v.dtype.is_floating_point)

    logger.info(f"\n      저장 경로  : {out}")
    logger.info(f"      파일 크기  : {size_mb:.1f} MB")
    logger.info(f"      총 파라미터: {n_total:,}  (backbone: {n_backbone:,})")
    logger.info(f"\n{'='*60}")
    logger.info(f"학습 config (sem_mae_em_pretrain.yaml) 설정:")
    logger.info(f"  student:")
    logger.info(f"    resume_from_teacher_chkpt: '{out}'")
    logger.info(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    backbone_sd, fmt = load_backbone_weights(args.ckpt)
    backbone = build_backbone(args)
    transfer_to_backbone(backbone, backbone_sd)
    dino_head, ibot_head = build_heads(args, backbone.embed_dim)
    save(backbone, dino_head, ibot_head, args.output)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pretrained 체크포인트 → DINOv3 SSLMetaArch 초기화 weight 변환",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True,
                   help="입력 체크포인트 (MAE .pth / DINOv3 .pth)")
    p.add_argument("--output", required=True,
                   help="출력 경로 (.pth)")

    # ── 아키텍처 (config student.* 와 일치) ───────────────────────────────
    p.add_argument("--arch", default="vit_large",
                   choices=["vit_small", "vit_base", "vit_large", "vit_so400m"])
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=448,
                   help="config crops.global_crops_size 와 일치")
    p.add_argument("--n_storage_tokens", type=int, default=4,
                   help="config student.n_storage_tokens")
    p.add_argument("--no_mask_k_bias", action="store_true",
                   help="student.mask_k_bias: false 인 경우 지정")
    p.add_argument("--norm_layer", default="layernormbf16",
                   choices=["layernorm", "layernormbf16"],
                   help="config student.norm_layer")

    # ── Head 크기 (config dino.* / ibot.* 와 일치) ───────────────────────
    p.add_argument("--dino_prototypes", type=int, default=65536,
                   help="config dino.head_n_prototypes")
    p.add_argument("--dino_hidden_dim", type=int, default=2048,
                   help="config dino.head_hidden_dim")
    p.add_argument("--ibot_prototypes", type=int, default=32768,
                   help="config ibot.head_n_prototypes")
    p.add_argument("--ibot_hidden_dim", type=int, default=2048,
                   help="config ibot.head_hidden_dim")
    p.add_argument("--head_bottleneck_dim", type=int, default=256,
                   help="config dino/ibot.head_bottleneck_dim (공통)")

    args = p.parse_args()
    args.mask_k_bias = not args.no_mask_k_bias
    return args


if __name__ == "__main__":
    main(parse_args())
