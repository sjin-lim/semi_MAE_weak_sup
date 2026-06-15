# Joint image+mask augmentation for labeled EM data.
#
# Uses torchvision.tv_tensors to keep mask in sync with geometric transforms
# while intensity transforms only touch the image.

import logging
import os
import sys
from typing import Tuple

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

# repo root import — PerImageNormalize 재사용 (semi_MAE 와 동일)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
try:
    from util.dataset_csv import PerImageNormalize
except Exception:
    PerImageNormalize = None

# Reuse the EM intensity + wave augmentation from existing augmentations module
from dinov3.data.augmentations import FusedEMIntensity, build_wave_modulation

logger = logging.getLogger("dinov3")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class JointGeometricAugmentation:
    """Image + mask 동기 geometric augmentation.

    - RandomResizedCrop (joint)
    - RandomHorizontalFlip / VerticalFlip (joint)
    - ElasticTransform (선택, image+mask joint)
    """

    def __init__(
        self,
        crop_size: int,
        crop_scale: tuple = (0.32, 1.0),
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        elastic: bool = False,
        elastic_alpha: float = 25.0,
        elastic_sigma: float = 8.0,
    ):
        transforms = [
            v2.RandomResizedCrop(
                crop_size,
                scale=crop_scale,
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=horizontal_flip_p),
            v2.RandomVerticalFlip(p=vertical_flip_p),
        ]
        if elastic:
            transforms.append(
                v2.RandomApply(
                    [v2.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma)],
                    p=0.3,
                )
            )
        self.transform = v2.Compose(transforms)

    def __call__(self, image: Image.Image, mask: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        # Wrap mask as tv_tensors.Mask so geometric ops auto-sync
        mask_tv = tv_tensors.Mask(mask)
        out_image, out_mask = self.transform(image, mask_tv)
        return out_image, out_mask.as_subclass(torch.Tensor)


class LabeledImageAugmentation:
    """Full augmentation pipeline for one labeled image.

    Multi-view (2 global crops) + per-image intensity augs.
    Mask only follows geometric transforms.

    Output:
        {
            "global_crops":     [tensor(3, H, W), tensor(3, H, W)],
            "global_crop_masks": [tensor(H, W),    tensor(H, W)],
            # local crops: image only (no mask needed, weak sup uses global only)
            "local_crops": [tensor(3, h, w), ...] * local_crops_number,
        }
    """

    def __init__(
        self,
        global_crops_size: int,
        local_crops_size: int,
        local_crops_number: int,
        global_crops_scale: tuple = (0.32, 1.0),
        local_crops_scale: tuple = (0.05, 0.32),
        horizontal_flips: bool = True,
        vertical_flips: bool = True,
        clahe: bool = False,
        intensity_aug_config: dict = None,
        wave_aug_config: dict = None,
        instance_norm: bool = True,
        mean: tuple = IMAGENET_DEFAULT_MEAN,
        std: tuple = IMAGENET_DEFAULT_STD,
    ):
        self.local_crops_number = local_crops_number

        self.joint_global = JointGeometricAugmentation(
            crop_size=global_crops_size,
            crop_scale=global_crops_scale,
            horizontal_flip_p=0.5 if horizontal_flips else 0.0,
            vertical_flip_p=0.5 if vertical_flips else 0.0,
        )
        self.geom_local = v2.Compose([
            v2.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            v2.RandomVerticalFlip(p=0.5 if vertical_flips else 0.0),
        ])

        # Intensity aug (image only). Config-driven.
        _ia_defaults = dict(
            clahe_p=0.5, gamma_p=0.5, poisson_p=0.3,
            disk_blur_p_global1=1.0, disk_blur_p_global2=0.1, disk_blur_p_local=0.5,
        )
        if intensity_aug_config is not None:
            _ia_defaults.update(dict(intensity_aug_config))

        self.intensity_g1 = FusedEMIntensity(
            clahe=clahe,
            clahe_p=_ia_defaults["clahe_p"],
            gamma_p=_ia_defaults["gamma_p"],
            poisson_p=_ia_defaults["poisson_p"],
            disk_blur_p=_ia_defaults["disk_blur_p_global1"],
        )
        self.intensity_g2 = FusedEMIntensity(
            clahe=clahe,
            clahe_p=_ia_defaults["clahe_p"],
            gamma_p=_ia_defaults["gamma_p"],
            poisson_p=_ia_defaults["poisson_p"],
            disk_blur_p=_ia_defaults["disk_blur_p_global2"],
        )
        self.intensity_local = FusedEMIntensity(
            clahe=clahe,
            clahe_p=_ia_defaults["clahe_p"],
            gamma_p=_ia_defaults["gamma_p"],
            poisson_p=_ia_defaults["poisson_p"],
            disk_blur_p=_ia_defaults["disk_blur_p_local"],
        )

        # Wave (ripple/반사) augmentation — global student view only.
        # 두 global crop 에 독립 wave realization → cross-view consistency 가
        # wave-invariance 학습 (clean teacher anchor 없이도 서로 다른 wave 라 유효).
        # 곱셈 + mean 보존 + 기하 불변 → mask 정합 영향 없음.
        self.wave_fns = [build_wave_modulation(wave_aug_config),
                         build_wave_modulation(wave_aug_config)]

        # Normalization
        if instance_norm and PerImageNormalize is not None:
            self.normalize = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                PerImageNormalize(),
            ])
        else:
            self.normalize = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=list(mean), std=list(std)),
            ])

    def __call__(self, image: Image.Image, mask: torch.Tensor) -> dict:
        output = {}

        # Two global views
        global_imgs = []
        global_masks = []
        intensity_fns = [self.intensity_g1, self.intensity_g2]
        for k in range(2):
            geo_img, geo_mask = self.joint_global(image, mask)
            int_img = intensity_fns[k](geo_img)
            if self.wave_fns[k] is not None:
                int_img = self.wave_fns[k](int_img)
            normalized = self.normalize(int_img)
            global_imgs.append(normalized)
            global_masks.append(geo_mask)

        output["global_crops"] = global_imgs
        output["global_crop_masks"] = global_masks

        # Local crops (image only)
        local_imgs = []
        for _ in range(self.local_crops_number):
            geo = self.geom_local(image)
            local_imgs.append(self.normalize(self.intensity_local(geo)))
        output["local_crops"] = local_imgs

        return output
