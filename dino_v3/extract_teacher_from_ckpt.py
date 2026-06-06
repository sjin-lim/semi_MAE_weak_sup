"""
FSDP sharded training checkpoint에서 teacher_checkpoint.pth 추출
(eval 자동저장이 안 된 경우 마지막 ckpt에서 직접 뽑아낼 때 사용)

사용:
    python extract_teacher_from_ckpt.py \
        --ckpt_dir /path/to/output_dir/ckpt/15999 \
        --output /path/to/teacher_checkpoint.pth
"""
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def init_single_process():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def extract_teacher(ckpt_dir: Path, output_path: Path):
    init_single_process()

    reader = dcp.FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata()

    # model_ema가 EMA teacher → teacher_checkpoint.pth로 저장될 가중치
    ema_keys = [k for k in metadata.state_dict_metadata.keys() if "model_ema" in k]
    if not ema_keys:
        raise RuntimeError(f"model_ema 키를 찾지 못함. metadata keys: {list(metadata.state_dict_metadata.keys())[:10]}")

    print(f"[load] model_ema 키 {len(ema_keys)}개 발견")

    # full tensor로 채울 빈 state_dict 템플릿 생성
    state_dict = {}
    for key in ema_keys:
        md = metadata.state_dict_metadata[key]
        state_dict[key] = torch.empty(md.size, dtype=md.properties.dtype)

    dcp.load(state_dict, storage_reader=reader)

    # "model.model_ema." or "model_ema." 접두사 제거 → teacher_checkpoint.pth 포맷과 일치
    teacher_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("model.model_ema.", "model_ema."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        teacher_sd[new_k] = v

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": teacher_sd}, output_path)
    print(f"[save] {output_path}  ({len(teacher_sd)} keys)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=Path, required=True,
                        help="FSDP sharded checkpoint 디렉토리 (예: output_dir/ckpt/15999)")
    parser.add_argument("--output", type=Path, required=True,
                        help="저장할 teacher_checkpoint.pth 경로")
    args = parser.parse_args()
    extract_teacher(args.ckpt_dir, args.output)
