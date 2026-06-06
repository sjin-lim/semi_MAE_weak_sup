# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, make_normalize_transform

# repo root (semi_MAE/) import — PerImageNormalize 재사용
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from util.dataset_csv import PerImageNormalize


class FusedEMIntensity:
    """EM 전용 intensity augmentation — PIL↔numpy 왕복 1회로 퓨전.

    기존에 CLAHE, Gamma, Poisson, DiskBlur 각각이 개별적으로
    PIL→numpy→PIL 변환을 수행해 crop 1개당 4회 왕복 (8회 malloc/free).
    이를 1회 왕복으로 통합해 메모리 단편화를 대폭 줄인다.

    10 crops × 128 batch = step당 5,120→1,280회로 malloc 호출 75% 감소.

    Args:
        clahe:          CLAHE 적용 여부.
        clahe_clip:     CLAHE clip limit.
        clahe_tile:     CLAHE tile size.
        clahe_p:        CLAHE 적용 확률.
        gamma_range:    Gamma correction 범위.
        gamma_p:        Gamma 적용 확률.
        poisson_range:  Poisson noise scale 범위.
        poisson_p:      Poisson noise 적용 확률.
        disk_blur_range: Disk blur radius 범위.
        disk_blur_p:    Disk blur 적용 확률.
    """

    def __init__(
        self,
        clahe: bool = False,
        clahe_clip: float = 2.0,
        clahe_tile: int = 8,
        clahe_p: float = 0.5,
        gamma_range: tuple = (0.7, 1.5),
        gamma_p: float = 0.5,
        poisson_range: tuple = (50.0, 200.0),
        poisson_p: float = 0.3,
        disk_blur_range: tuple = (0.5, 2.5),
        disk_blur_p: float = 0.5,
    ):
        self.use_clahe = clahe
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_tile, clahe_tile),
        ) if clahe else None
        self.clahe_p = clahe_p
        self.gamma_range = gamma_range
        self.gamma_p = gamma_p
        self.poisson_range = poisson_range
        self.poisson_p = poisson_p
        self.disk_blur_range = disk_blur_range
        self.disk_blur_p = disk_blur_p

    def __call__(self, img: Image.Image) -> Image.Image:
        # PIL → numpy 1회
        arr = np.array(img)

        # 1) CLAHE (grayscale 변환 → 적용 → RGB 복원)
        if self._clahe is not None and np.random.random() < self.clahe_p:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
            enhanced = self._clahe.apply(gray)
            arr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB) if arr.ndim == 3 else enhanced

        # 2) Gamma correction
        if np.random.random() < self.gamma_p:
            gamma = np.random.uniform(*self.gamma_range)
            arr_f = arr.astype(np.float32) * (1.0 / 255.0)
            np.clip(arr_f, 0, 1, out=arr_f)
            np.power(arr_f, gamma, out=arr_f)
            arr_f *= 255.0
            arr = arr_f.astype(np.uint8)
            del arr_f

        # 3) Poisson noise
        if np.random.random() < self.poisson_p:
            scale = np.random.uniform(*self.poisson_range)
            lam = arr.astype(np.float32) * (scale / 255.0)
            np.clip(lam, 0, None, out=lam)
            noisy = np.random.poisson(lam)
            del lam
            noisy = (noisy * (255.0 / scale))
            np.clip(noisy, 0, 255, out=noisy)
            arr = noisy.astype(np.uint8)
            del noisy

        # 4) Disk blur (TEM defocus)
        if np.random.random() < self.disk_blur_p:
            radius = np.random.uniform(*self.disk_blur_range)
            kernel_size = int(np.ceil(radius) * 2 + 1)
            if kernel_size >= 3:
                center = kernel_size // 2
                y, x = np.ogrid[:kernel_size, :kernel_size]
                mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
                kernel = mask.astype(np.float32)
                kernel /= kernel.sum()
                arr = cv2.filter2D(arr, -1, kernel)

        # numpy → PIL 1회
        # 주의: img.close() 하지 않음. Compose 파이프라인에서 resize_global이
        # nn.Identity()이면 img == im1_base(호출자의 원본)이므로
        # 여기서 close하면 호출자가 closed image를 teacher/gram crop에 사용하게 됨.
        # 해제는 DataAugmentationDINO.__call__ 에서 일괄 처리.
        result = Image.fromarray(arr, mode=img.mode)
        del arr
        return result

    def __repr__(self):
        parts = []
        if self.use_clahe:
            parts.append(f"CLAHE(p={self.clahe_p})")
        parts.append(f"Gamma(p={self.gamma_p})")
        parts.append(f"Poisson(p={self.poisson_p})")
        parts.append(f"DiskBlur(p={self.disk_blur_p})")
        return f"FusedEMIntensity({', '.join(parts)})"


logger = logging.getLogger("dinov3")

# ============================================================
# Augmentation 플러그인 리스트
# 여기서 항목을 추가/제거하면 모든 crop 파이프라인에 반영됩니다.
# ============================================================

def _build_geometric_extra() -> list:
    """geometric augmentation 리스트 동적 생성.

    호출할 때마다 새 인스턴스를 생성한다.
    ElasticTransform 등이 내부에 displacement field 텐서를 캐시하므로
    여러 Compose에서 같은 인스턴스를 공유하면 참조가 누적될 수 있다.
    """
    return [
        v2.RandomVerticalFlip(p=0.5),                        # TEM/SEM 방향성 없음
        v2.RandomApply(                                      # TEM drift/distortion 모사
            [v2.ElasticTransform(alpha=25.0, sigma=8.0)],    # alpha 50→25, sigma 5→8 (미세 구조 보존)
            p=0.3,
        ),
        # v2.RandomRotation(degrees=90),                     # 필요 시 주석 해제
        # v2.RandomAffine(degrees=0, shear=(-5, 5, -5, 5)),  # 필요 시 주석 해제
    ]


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        instance_norm=False,
        clahe=False,
        intensity_aug_config=None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self.instance_norm = instance_norm
        self.clahe = clahe

        # config 기반 intensity augmentation 구성
        # 각 파이프라인마다 별도 인스턴스 생성 — 공유 시 cv2/PIL 내부 상태 누적 방지

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info(f"instance_norm (EM): {instance_norm}")
        logger.info(f"clahe: {clahe}")
        logger.info(f"EM_GEOMETRIC_EXTRA: {[type(t).__name__ for t in _build_geometric_extra()]}")
        logger.info(f"EM_INTENSITY: {FusedEMIntensity(clahe=clahe)}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop + flip + EM geometric extras
        # 각 파이프라인마다 별도 인스턴스 생성 — ElasticTransform 내부 캐시 공유 방지
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
            + _build_geometric_extra()
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for DiskBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
            + _build_geometric_extra()
        )

        # ── EM 전용 intensity pipeline (퓨전) ──────────────────────────────
        # CLAHE + Gamma + Poisson + DiskBlur를 단일 클래스에서 처리.
        # PIL↔numpy 왕복이 4회→1회로 감소 → malloc 단편화 75% 감소.
        #
        # global crop 1: disk blur p=1.0 (강한 blur — view 다양성 확보)
        # global crop 2: disk blur p=0.1 (약한 blur)
        # local crop:    disk blur p=0.5
        # 각 파이프라인마다 별도 인스턴스 생성 (cv2 CLAHE 내부 버퍼 공유 방지)

        # ── intensity augmentation 설정 (ablation 용 config-driven) ──────────
        # intensity_aug_config 에 일부 key 만 지정해도 나머지는 default 유지
        _ia_defaults = dict(
            clahe_p=0.5,
            gamma_p=0.5,
            poisson_p=0.3,
            disk_blur_p_global1=1.0,
            disk_blur_p_global2=0.1,
            disk_blur_p_local=0.5,
        )
        if intensity_aug_config is not None:
            _ia_defaults.update(dict(intensity_aug_config))
        logger.info(f"intensity_aug_config (effective): {_ia_defaults}")

        global_transfo1 = FusedEMIntensity(
            clahe=clahe,
            clahe_p=_ia_defaults["clahe_p"],
            gamma_p=_ia_defaults["gamma_p"],
            poisson_p=_ia_defaults["poisson_p"],
            disk_blur_p=_ia_defaults["disk_blur_p_global1"],
        )
        global_transfo2 = FusedEMIntensity(
            clahe=clahe,
            clahe_p=_ia_defaults["clahe_p"],
            gamma_p=_ia_defaults["gamma_p"],
            poisson_p=_ia_defaults["poisson_p"],
            disk_blur_p=_ia_defaults["disk_blur_p_global2"],
        )
        local_transfo = FusedEMIntensity(
            clahe=clahe,
            clahe_p=_ia_defaults["clahe_p"],
            gamma_p=_ia_defaults["gamma_p"],
            poisson_p=_ia_defaults["poisson_p"],
            disk_blur_p=_ia_defaults["disk_blur_p_local"],
        )

        # normalization: instance_norm(EM 권장) 또는 ImageNet mean/std
        if instance_norm:
            # Per-image normalization (MAE 의 PerImageNormalize 재사용)
            self.normalize = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    PerImageNormalize(),
                ]
            )
        else:
            self.normalize = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    make_normalize_transform(mean=mean, std=std),
                ]
            )

        self.global_transfo1 = v2.Compose([resize_global, global_transfo1, self.normalize])
        self.global_transfo2 = v2.Compose([resize_global, global_transfo2, self.normalize])
        self.local_transfo = v2.Compose([local_transfo, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = []
            for _ in range(self.local_crops_number):
                local_img = self.geometric_augmentation_local(image)
                local_crops.append(self.local_transfo(local_img))
                # geometric augmentation이 PIL을 반환하는 경우 즉시 해제
                if hasattr(local_img, 'close'):
                    local_img.close()
            output["local_crops"] = local_crops
            output["offsets"] = ()

        # ── 중간 PIL 이미지 해제 ──────────────────────────────
        # im1_base, im2_base는 geometric augmentation의 PIL 출력.
        # 모든 crop이 tensor로 변환된 후에는 불필요 → 즉시 close.
        # PIL Image는 내부에 circular reference를 가지므로
        # close() 없이는 GC까지 메모리가 유지된다.
        if hasattr(im1_base, 'close'):
            im1_base.close()
        if hasattr(im2_base, 'close'):
            im2_base.close()

        return output
