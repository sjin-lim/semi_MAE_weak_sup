# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import random

import numpy as np
import torch


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(np.where(~m2)[0], size=num_masking_patches - m2.sum(), replace=False)
        m2[to_add] = True
        return m2.reshape(shape)


class InformationAwareMaskingGenerator:
    """정보량 기반 마스킹 — 저정보(균일 배경) 패치를 우선 마스킹.

    EM 이미지 특성:
      - 패치 대부분이 비슷한 회색 배경
      - 구조적 의미가 있는 패치(결정립계, 전위 등)는 소수
      - 랜덤 마스킹 시 의미 있는 패치가 가려져 복원 단서 소실

    동작 원리:
      1. 이미지를 패치 그리드로 분할
      2. 각 패치의 픽셀 표준편차 계산 (정보량 지표)
      3. std 낮은(정보량 적은) 패치에 높은 마스킹 확률 부여
      4. 확률적 샘플링으로 마스크 생성

    효과:
      - 구조적 패치(edge)가 context로 보존됨
      - student는 "어떤 구조 옆의 배경"을 맞추는 과제를 풀게 됨
      - edge representation이 자연스럽게 정교해짐

    Args:
        input_size:   패치 그리드 크기 (H_patches, W_patches).
        temperature:  확률 분포 날카로움. 낮을수록 저정보 패치에 집중.
                      0이면 완전히 std 역순 결정론적, 높을수록 랜덤에 가까움.
        fallback_generator: 이미지 정보 없을 때 사용할 랜덤 마스킹 generator.
    """

    def __init__(
        self,
        input_size,
        temperature: float = 0.5,
        fallback_generator: MaskingGenerator = None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.temperature = temperature
        self.fallback = fallback_generator

    def compute_patch_std(self, image_tensor: torch.Tensor) -> np.ndarray:
        """이미지 텐서에서 패치별 표준편차 계산.

        EM shot noise 대응:
          패치 내부를 3×3 average pooling으로 한 번 스무딩한 뒤 std를 계산.
          구조 없이 노이즈만 있는 패치는 스무딩 후 std가 급감하지만,
          실제 구조(edge)가 있는 패치는 스무딩 후에도 std가 유지됨.

        Args:
            image_tensor: (C, H, W) 정규화된 이미지 텐서.

        Returns:
            (H_patches, W_patches) 패치별 std 배열.
        """
        C, H, W = image_tensor.shape
        patch_h = H // self.height
        patch_w = W // self.width

        # shot noise 억제: 3×3 average pooling (padding=1로 크기 유지)
        # collate 에서 넘어오는 텐서는 이미 float32 — 불필요한 복사 방지
        img = image_tensor if image_tensor.is_floating_point() else image_tensor.float()
        smoothed = torch.nn.functional.avg_pool2d(
            img.unsqueeze(0),  # (1, C, H, W)
            kernel_size=3, stride=1, padding=1,
        ).squeeze(0)  # (C, H, W)

        # (C, H, W) → (C, nH, patch_h, nW, patch_w) → (nH, nW, C*patch_h*patch_w)
        patches = smoothed.reshape(C, self.height, patch_h, self.width, patch_w)
        patches = patches.permute(1, 3, 0, 2, 4).reshape(
            self.height, self.width, -1
        )  # (nH, nW, C*pH*pW)

        # 패치별 std — float32로 계산 (bf16 정밀도 부족)
        patch_std = patches.std(dim=-1).numpy()  # (nH, nW)
        return patch_std

    def __call__(self, num_masking_patches: int = 0, image_tensor: torch.Tensor = None):
        """마스크 생성.

        Args:
            num_masking_patches: 마스킹할 패치 수.
            image_tensor:        (C, H, W) 이미지 텐서. None이면 랜덤 fallback.

        Returns:
            (H_patches, W_patches) bool 마스크.
        """
        if num_masking_patches == 0:
            return np.zeros((self.height, self.width), dtype=bool)

        # 이미지 정보 없으면 fallback
        if image_tensor is None:
            if self.fallback is not None:
                return self.fallback(num_masking_patches)
            return self._uniform_mask(num_masking_patches)

        patch_std = self.compute_patch_std(image_tensor)

        # 정보량이 낮은 패치(std↓)에 높은 마스킹 확률 부여
        # inverse std → 높은 값 = 마스킹 우선
        inv_std = 1.0 / (patch_std + 1e-6)

        # temperature로 분포 조절
        # temperature가 높을수록 uniform에 가까움 (랜덤성 유지)
        # temperature가 낮을수록 저정보 패치에 집중
        if self.temperature > 0:
            logits = inv_std / self.temperature
        else:
            logits = inv_std

        # softmax → 확률 분포
        logits_flat = logits.flatten()
        logits_flat = logits_flat - logits_flat.max()  # numerical stability
        probs = np.exp(logits_flat)
        probs = np.maximum(probs, 1e-8)  # 확률 하한 — exp underflow 방지
        probs = probs / probs.sum()

        # 확률적 비복원 샘플링
        num_masking_patches = min(num_masking_patches, self.num_patches)
        chosen = np.random.choice(
            self.num_patches,
            size=num_masking_patches,
            replace=False,
            p=probs,
        )

        mask = np.zeros(self.num_patches, dtype=bool)
        mask[chosen] = True
        return mask.reshape(self.height, self.width)

    def _uniform_mask(self, num_masking_patches: int) -> np.ndarray:
        """이미지 정보 없을 때 균등 랜덤 마스킹."""
        num_masking_patches = min(num_masking_patches, self.num_patches)
        chosen = np.random.choice(self.num_patches, size=num_masking_patches, replace=False)
        mask = np.zeros(self.num_patches, dtype=bool)
        mask[chosen] = True
        return mask.reshape(self.height, self.width)
