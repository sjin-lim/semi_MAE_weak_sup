"""
HDF5 기반 Dataset 클래스.

preprocess_to_h5.py 로 생성된 .h5 파일을 읽어 학습에 사용합니다.

로딩 모드 (preload 인수):
  preload=True  : 초기화 시 전체 이미지를 torch shared memory 로 로드.
                  모든 DataLoader 워커가 동일한 물리 메모리를 참조하므로
                  RAM = dataset_size × 1 (워커 수와 무관).
                  100K × 448×448×3 uint8 ≈ 60 GB — 서버 RAM이 충분할 때 사용.
  preload=False : 워커별 h5py lazy open (기존 동작).
                  RAM = 워커당 독립 h5 버퍼 소비.

HDF5 구조 (preprocess_to_h5.py 출력):
  /images         (N, H, W, 3) uint8  — 메인 이미지 (사전 리사이즈)
                  또는 (N,) vlen uint8  — raw bytes 모드
  /filtered       (N, ...)            — (옵션) 필터 이미지
  attrs: n_images, resize, has_filtered
"""

import io
import logging
import random

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class H5ImageDataset(Dataset):
    """HDF5 파일에서 이미지를 로드하는 Dataset.

    Args:
        h5_path: preprocess_to_h5.py 로 생성된 HDF5 파일 경로.
        use_filtered_target: True이면 (image, filtered_image) 쌍 반환.
        transform: torchvision transforms.
        preload: True이면 전체 데이터를 shared memory로 미리 올림.
                 False이면 워커별 h5py lazy open (기존 동작).
    """

    def __init__(self, h5_path: str,
                 use_filtered_target: bool = False,
                 transform=None,
                 preload: bool = False):
        self.h5_path = h5_path
        self.use_filtered_target = use_filtered_target
        self.transform = transform
        self.preload = preload
        self._images = None
        self._filtered = None

        # 메타데이터 읽기
        with h5py.File(h5_path, 'r') as f:
            self._len = int(f.attrs.get('n_images', f['images'].shape[0]))
            self._is_vlen = f['images'].dtype == h5py.vlen_dtype(np.dtype('uint8'))
            has_filtered = bool(f.attrs.get('has_filtered', 'filtered' in f))

            if use_filtered_target and not has_filtered:
                raise ValueError(
                    f"HDF5 파일에 '/filtered' 데이터셋이 없습니다: {h5_path}\n"
                    "preprocess_to_h5.py 실행 시 --filtered_col 을 지정했는지 확인하세요."
                )

            if preload:
                self._preload(f, has_filtered)

        # lazy open용 파일 핸들 (preload=False 시 사용)
        self._file = None

    # ------------------------------------------------------------------
    # Shared memory preload
    # ------------------------------------------------------------------

    def _preload(self, f: h5py.File, has_filtered: bool):
        """전체 이미지를 shared memory tensor로 로드.

        DataLoader fork 이후 모든 워커가 동일한 물리 메모리 페이지를 읽으므로
        RAM 사용량이 워커 수에 비례해 늘어나지 않는다.
        vlen(raw bytes) 모드는 가변 길이라 shared_memory 불가 → 경고 후 fallback.
        """
        if self._is_vlen:
            logger.warning(
                "[H5ImageDataset] vlen(raw bytes) 모드는 preload를 지원하지 않습니다. "
                "lazy open으로 fallback합니다. "
                "h5 파일 생성 시 --resize 옵션으로 고정 크기 uint8 배열로 저장하면 preload가 가능합니다."
            )
            self.preload = False
            return

        logger.info(f"[H5ImageDataset] preload 시작: {self.h5_path}")
        images_np = f['images'][:]                   # (N, H, W, 3) uint8 전체 로드
        self._images = torch.from_numpy(images_np)
        self._images.share_memory_()                 # fork 후 모든 워커가 공유
        logger.info(f"[H5ImageDataset] images shared memory: {self._images.nbytes/1e9:.1f} GB")

        if has_filtered and self.use_filtered_target:
            filtered_np = f['filtered'][:]
            self._filtered = torch.from_numpy(filtered_np)
            self._filtered.share_memory_()
            logger.info(f"[H5ImageDataset] filtered shared memory: {self._filtered.nbytes/1e9:.1f} GB")

    # ------------------------------------------------------------------
    # lazy open (preload=False)
    # ------------------------------------------------------------------

    def __len__(self):
        return self._len

    def _get_file(self):
        """워커별 lazy open. fork-safe를 위해 __init__에서 열지 않음."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._file

    @staticmethod
    def _arr_to_pil(arr) -> Image.Image:
        """uint8 numpy/tensor 배열 또는 raw bytes → PIL RGB Image."""
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        if arr.ndim == 1:
            # vlen raw bytes 모드
            img = Image.open(io.BytesIO(arr.tobytes()))
        else:
            # (H, W, 3) uint8 배열
            img = Image.fromarray(arr, mode='RGB')

        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        # preload 모드: shared memory에서 직접 읽기 (h5py I/O 없음)
        if self.preload and self._images is not None:
            arr = self._images[idx]
            img = self._arr_to_pil(arr)

            if self.use_filtered_target and self._filtered is not None:
                flt = self._arr_to_pil(self._filtered[idx])
                seed = random.randint(0, 2 ** 31)
                random.seed(seed); torch.manual_seed(seed)
                img_tensor = self.transform(img)
                random.seed(seed); torch.manual_seed(seed)
                flt_tensor = self.transform(flt)
                return img_tensor, flt_tensor

            if self.transform is not None:
                img = self.transform(img)
            return img, 0

        # lazy open 모드
        f = self._get_file()
        arr = f['images'][idx]

        if self.use_filtered_target:
            flt_arr = f['filtered'][idx]
            img = self._arr_to_pil(arr)
            flt = self._arr_to_pil(flt_arr)
            seed = random.randint(0, 2 ** 31)
            random.seed(seed); torch.manual_seed(seed)
            img_tensor = self.transform(img)
            random.seed(seed); torch.manual_seed(seed)
            flt_tensor = self.transform(flt)
            return img_tensor, flt_tensor

        img = self._arr_to_pil(arr)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass