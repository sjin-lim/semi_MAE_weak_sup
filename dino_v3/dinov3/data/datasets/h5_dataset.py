# Copyright (c) Meta Platforms, Inc. and affiliates.
# (Custom addition for semi_MAE project)
#
# HDF5 / CSV / ImageFolder 통합 Dataset — DINOv3 호환.
# util.dataset_h5 / util.dataset_csv 에 대한 module-level 의존성을 제거하고
# h5 로직을 직접 구현해 circular import 를 방지합니다.

import io
import logging
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# repo root (semi_MAE/) — csv 모드에서만 lazy import 시 사용
_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../../"))


class SemiMAEDataset(Dataset):
    """DINOv3 호환 Dataset.

    H5 / CSV / ImageFolder 세 가지 소스를 지원합니다.
    util.dataset_h5 를 import 하지 않고 h5 로직을 직접 구현해
    circular import 문제를 근본적으로 해결합니다.

    Args:
        h5_path:             preprocess_to_h5.py 로 생성된 .h5 파일 경로.
        csv_path:            이미지 경로 DataFrame (.pkl / .csv / .parquet).
        root:                ImageFolder 루트 디렉토리 (h5/csv 없을 때 fallback).
        filepath_col:        CSV 내 이미지 경로 컬럼명 (기본: 'filepath').
        use_filtered_target: True 이면 __getitem__ 이 (aug(img), aug(filtered)) 반환.
        filtered_col:        CSV 내 필터 이미지 경로 컬럼명.
        transforms:          DataAugmentationDINO 인스턴스 또는 callable.
        preload:             True 이면 전체 h5 데이터를 shared memory 로 미리 로드.
                             RAM = dataset_size × 1 (워커 수 무관).
                             vlen(raw bytes) 모드에서는 자동으로 False 로 fallback.
    """

    def __init__(
        self,
        *,
        h5_path: str = "",
        csv_path: str = "",
        root: str = "",
        filepath_col: str = "filepath",
        use_filtered_target: bool = False,
        filtered_col: str = "filtered_filepath",
        transforms=None,
        preload: bool = False,
    ):
        self.transforms = transforms
        self.use_filtered_target = use_filtered_target
        self._mode = None   # 'h5' | 'csv' | 'folder'

        # shared memory tensors (preload=True, h5 모드)
        self._images = None
        self._filtered = None
        # lazy open 핸들 (preload=False, h5 모드)
        self._h5_path = None
        self._h5_file = None
        self._is_vlen = False

        if h5_path:
            self._mode = "h5"
            self._h5_path = h5_path
            self._init_h5(h5_path, use_filtered_target, preload)

        elif csv_path:
            self._mode = "csv"
            self._init_csv(csv_path, filepath_col, filtered_col, use_filtered_target)

        elif root:
            self._mode = "folder"
            import torchvision.datasets as tvd
            self._inner = tvd.ImageFolder(root)

        else:
            raise ValueError("h5_path, csv_path, 또는 root 중 하나를 지정해야 합니다.")

    # ------------------------------------------------------------------
    # H5 초기화
    # ------------------------------------------------------------------

    def _init_h5(self, h5_path: str, use_filtered_target: bool, preload: bool):
        with h5py.File(h5_path, 'r') as f:
            self._len = int(f.attrs.get('n_images', f['images'].shape[0]))
            self._is_vlen = f['images'].dtype == h5py.vlen_dtype(np.dtype('uint8'))
            has_filtered = bool(f.attrs.get('has_filtered', 'filtered' in f))

            if use_filtered_target and not has_filtered:
                raise ValueError(
                    f"HDF5 파일에 '/filtered' 데이터셋이 없습니다: {h5_path}"
                )

            if preload:
                if self._is_vlen:
                    logger.warning(
                        "[SemiMAEDataset] vlen(raw bytes) 모드는 preload 미지원 → lazy open으로 fallback. "
                        "h5 생성 시 --resize 옵션으로 고정 크기 uint8 배열로 저장하면 preload 가능."
                    )
                else:
                    logger.info(f"[SemiMAEDataset] preload 시작: {h5_path}")
                    images_np = f['images'][:]
                    self._images = torch.from_numpy(images_np)
                    self._images.share_memory_()
                    logger.info(f"[SemiMAEDataset] images shared memory: {self._images.nbytes/1e9:.1f} GB")

                    if has_filtered and use_filtered_target:
                        filtered_np = f['filtered'][:]
                        self._filtered = torch.from_numpy(filtered_np)
                        self._filtered.share_memory_()
                        logger.info(f"[SemiMAEDataset] filtered shared memory: {self._filtered.nbytes/1e9:.1f} GB")

    # ------------------------------------------------------------------
    # CSV 초기화 (lazy import — csv 모드에서만 실행)
    # ------------------------------------------------------------------

    def _init_csv(self, csv_path, filepath_col, filtered_col, use_filtered_target):
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from util.dataset_csv import CSVImageDataset, load_dataframe
        df = load_dataframe(csv_path)
        self._inner = CSVImageDataset(
            df,
            filepath_col=filepath_col,
            filtered_col=filtered_col,
            use_filtered_target=use_filtered_target,
            transform=None,
        )
        self._len = len(self._inner)

    # ------------------------------------------------------------------
    # H5 helpers
    # ------------------------------------------------------------------

    def _get_h5_file(self):
        """워커별 lazy open (preload=False 시 사용).

        rdcc_nbytes: 워커당 h5py 청크 캐시 크기.
        랜덤 접근 시 캐시 히트율을 높여 disk I/O를 줄인다.
        워커 4개 × 2GB = 8GB — 서버 RAM에 맞게 조정 가능.
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(
                self._h5_path, 'r',
                swmr=True,
                rdcc_nbytes=512 * 1024 ** 2,   # 워커당 512MB 청크 캐시
                rdcc_nslots=1_000_003,          # 소수 → 해시 충돌 최소화
            )
        return self._h5_file

    @staticmethod
    def _arr_to_pil(arr) -> Image.Image:
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        if arr.ndim == 1:
            # vlen raw bytes 모드
            img = Image.open(io.BytesIO(arr.tobytes()))
            img.load()  # lazy load 해제 → BytesIO 참조 즉시 끊김
        elif arr.shape[-1] == 1:
            # (H, W, 1) grayscale → PIL 'L'
            img = Image.fromarray(arr[..., 0], mode='L')
        else:
            # (H, W, 3) RGB
            img = Image.fromarray(arr, mode='RGB')
        # 항상 RGB로 변환 (모델 in_chans=3)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    # ------------------------------------------------------------------
    # PIL 취득
    # ------------------------------------------------------------------

    def _get_pil(self, index):
        if self._mode == "h5":
            if self._images is not None:
                return self._arr_to_pil(self._images[index])
            f = self._get_h5_file()
            return self._arr_to_pil(f['images'][index])

        elif self._mode == "csv":
            from util.dataset_csv import CSVImageDataset
            return CSVImageDataset._load_rgb(self._inner.paths[index])

        else:  # folder
            path, _ = self._inner.samples[index]
            img = Image.open(path)
            return img.convert('RGB') if img.mode != 'RGB' else img

    def _get_filtered_pil(self, index):
        if self._mode == "h5":
            if self._filtered is not None:
                return self._arr_to_pil(self._filtered[index])
            f = self._get_h5_file()
            return self._arr_to_pil(f['filtered'][index])
        elif self._mode == "csv":
            from util.dataset_csv import CSVImageDataset
            return CSVImageDataset._load_rgb(self._inner.filtered_paths[index])
        else:
            raise NotImplementedError("ImageFolder mode does not support filtered targets.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        if self._mode in ("h5",):
            return self._len
        return len(self._inner)

    def __getitem__(self, index):
        img = self._get_pil(index)

        if self.use_filtered_target:
            filtered = self._get_filtered_pil(index)
            if self.transforms is not None:
                return (self.transforms(img), self.transforms(filtered)), 0
            return (img, filtered), 0

        if self.transforms is not None:
            return self.transforms(img), 0
        return img, 0

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# WebDataset 기반 Dataset (sequential read, 병목 없음)
# ─────────────────────────────────────────────────────────────────────────────

class SemiMAEWebDataset(torch.utils.data.IterableDataset):
    """WebDataset(tar shard) 기반 IterableDataset.

    HDF5 random access 대비 sequential read라 I/O 병목이 없음.
    convert_to_wds.py 로 생성된 tar 파일 디렉토리를 입력으로 받음.

    멀티 GPU:  wds.split_by_node() → 각 rank가 서로 다른 샤드 담당
    멀티 워커: wds.split_by_worker() → 워커 간 샤드 분배
    셔플링:    샤드 순서 셔플 + buffer shuffle (샘플 단위)

    Args:
        path:         tar 파일이 있는 디렉토리 경로.
        transforms:   DataAugmentationDINO 인스턴스.
        buffer_size:  sample-level shuffle 버퍼 크기.
                      데이터셋 크기의 5~10% 권장 (기본 5000).
        shardshuffle: 샤드 순서 셔플 여부 (학습 시 True).
        min_std:      이미지 픽셀 표준편차 최소 임계값.
                      이보다 낮은 이미지(균일 배경/노이즈)는 스킵.
                      iBOT loss 안정화에 중요. 0이면 필터링 비활성화.
    """

    def __init__(self, path: str, transforms=None,
                 buffer_size: int = 5000, shardshuffle: bool = True,
                 min_std: float = 5.0,
                 pipeline_restart_interval: int = 10000):
        try:
            import webdataset  # noqa: F401
        except ImportError:
            raise ImportError("pip install webdataset")

        import glob
        self.urls = sorted(glob.glob(os.path.join(path, "*.tar")))
        if not self.urls:
            raise FileNotFoundError(f"tar 파일을 찾을 수 없습니다: {path}/*.tar")
        logger.info(f"[SemiMAEWebDataset] {len(self.urls)}개 샤드 발견: {path}")

        self.transforms = transforms
        self.buffer_size = buffer_size
        self.shardshuffle = shardshuffle
        self.min_std = min_std
        self._skip_count = 0
        self._restart_interval = pipeline_restart_interval

    def _build_pipeline(self):
        """WebDataset 파이프라인 1개 생성. __iter__에서 주기적으로 재생성."""
        import webdataset as wds

        def _handler(exn):
            """traceback-safe handler — 로컬 프레임 참조 즉시 정리."""
            logger.warning(f"[WebDataset] 샘플 스킵: {type(exn).__name__}: {exn}")
            return True

        return (
            wds.WebDataset(
                self.urls,
                resampled=True,
                nodesplitter=wds.split_by_node,
                handler=_handler,
            )
            .shuffle(self.buffer_size)
            .map(self._process, handler=_handler)
        )

    def __iter__(self):
        import gc
        import ctypes

        # malloc_trim 준비 — worker에서 힙 단편화 방지
        try:
            _libc = ctypes.CDLL("libc.so.6")
            _has_malloc_trim = True
        except OSError:
            _has_malloc_trim = False

        _GC_EVERY = 200  # 200 샘플마다 GC + malloc_trim

        # ── pipeline 주기적 재생성 ──────────────────────────────
        # resampled=True → worker가 영원히 살아있으므로,
        # pipeline 내부(tar handle, shuffle buffer bytes, webdataset 상태)에
        # 미세하게 누적되는 참조/단편화를 주기적으로 완전 해제.
        while True:
            pipeline = self._build_pipeline()
            sample_count = 0
            for sample in pipeline:
                yield sample
                sample_count += 1

                # ── 주기적 GC + malloc_trim ─────────────────────
                # augmentation이 매 샘플마다 대용량 임시 배열을 할당/해제.
                # glibc가 freed memory를 OS에 안 돌려줘서 RSS 성장.
                # 200샘플마다 강제로 순환참조 수거 + 힙 compaction.
                if sample_count % _GC_EVERY == 0:
                    gc.collect()
                    if _has_malloc_trim:
                        _libc.malloc_trim(0)

                if sample_count >= self._restart_interval:
                    break

            # pipeline 파기 → shuffle buffer, tar handle 등 완전 해제
            del pipeline
            gc.collect()
            if _has_malloc_trim:
                _libc.malloc_trim(0)
            logger.info(f"[SemiMAEWebDataset] pipeline 재생성 "
                        f"(매 {self._restart_interval} 샘플)")

    def _process(self, sample: dict):
        img = None
        try:
            # JPEG (primary) — 원본 해상도 + 압축으로 I/O/메모리 효율적
            raw = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
            if raw is not None:
                buf = io.BytesIO(raw) if isinstance(raw, bytes) else None
                img = Image.open(buf) if buf is not None else raw
                img.load()  # lazy load 해제 → BytesIO 참조 즉시 끊김
                del buf, raw
            elif "npy" in sample:
                # NPY fallback — 기존 샤드 호환
                raw = sample["npy"]
                arr = np.load(io.BytesIO(raw)) if isinstance(raw, bytes) else raw
                img = Image.fromarray(arr, mode='L')
                del arr, raw
            else:
                raise ValueError(f"샘플에 이미지 키가 없음: {list(sample.keys())}")

            # sample dict의 대용량 바이너리 참조를 미리 정리
            sample.clear()

            # 저분산 이미지 필터링 — 균일 배경/노이즈만 있는 무의미 패치 제거
            # 원본 해상도 전체를 numpy 변환하면 수십 MB 할당 → 힙 단편화.
            # 128×128 썸네일로 축소 후 std 계산 — 판별 정확도 동일, 메모리 99% 절감.
            if self.min_std > 0:
                thumb = img.copy()
                thumb.thumbnail((128, 128))
                pixel_arr = np.array(thumb)
                std_val = pixel_arr.std()
                del pixel_arr, thumb
                if std_val < self.min_std:
                    self._skip_count += 1
                    if self._skip_count % 500 == 1:
                        logger.info(f"[SemiMAEWebDataset] 저분산 이미지 스킵 누적 {self._skip_count}건 "
                                    f"(std={std_val:.1f} < {self.min_std})")
                    img.close()
                    raise ValueError(f"저분산 이미지 스킵 (std={std_val:.1f})")

            if img.mode != "RGB":
                rgb = img.convert("RGB")
                img.close()
                img = rgb

            if self.transforms is not None:
                result = self.transforms(img), 0
                img.close()  # transform 완료 → PIL 원본 즉시 해제
                return result
            return img, 0
        except Exception:
            if img is not None:
                img.close()  # 에러 경로에서도 PIL 이미지 반드시 해제
            raise
