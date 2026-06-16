# Copyright (c) 2026.
#
# EM classification 용 augmentation 모음.
#
# few-shot/linear 분류기를 frozen DINO feature 위에서 학습할 때, support(train)
# 이미지에 augmentation 을 적용해 여러 view 의 feature 를 뽑아 학습 데이터를
# 늘리는 용도 (feature-space data augmentation). query/test 는 clean 으로 둔다.
#
# ★ PerImageNormalize 주의 ────────────────────────────────────────────────
#   학습이 instance_norm(per-image zero-mean/unit-std)으로 진행되므로, eval 도
#   PerImageNormalize 로 끝낸다. 이때 **선형 밝기/대비(affine) 변화는 per-image
#   표준화로 완전히 상쇄**되어 augmentation 효과가 사라진다.
#   → "밝기"는 선형 scale 이 아니라 감마(GammaJitter, 비선형)로 넣어야 살아남는다.
#   → 노이즈/검은점/기하 변환은 instance-norm 후에도 효과가 남는다.
#
# 변환 순서:
#   ToImage → RandomResizedCrop(~0.8) → Rot90 → H/V Flip      (기하, uint8)
#   → ToDtype(float, scale) → Gamma → GaussianNoise → BlackDots
#   → PerImageNormalize                                       (강도, float)

import torch
from torchvision.transforms import v2


class PerImageNormalize:
    """학습과 동일한 per-image(instance) 정규화. (C,H,W) → zero-mean/unit-std."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True)
        return (tensor - mean) / (std + 1e-6)


class RandomRot90:
    """90도 단위 회전 (0/90/180/270). EM 은 canonical orientation 이 없어 안전."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(())) >= self.p:
            return x
        k = int(torch.randint(0, 4, ()))
        return torch.rot90(x, k, dims=[-2, -1])


class GammaJitter:
    """비선형 밝기. x in [0,1] → x**gamma, gamma=exp(U(-log_range, log_range)).

    per-image 표준화 후에도 살아남도록 선형 scale 대신 감마를 쓴다.
    log_range=0.4 → gamma ≈ [0.67, 1.49].
    """

    def __init__(self, p: float = 0.5, log_range: float = 0.4):
        self.p = p
        self.log_range = log_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(())) >= self.p:
            return x
        g = float(torch.empty(()).uniform_(-self.log_range, self.log_range).exp())
        return x.clamp(0, 1).pow(g)


class GaussianNoise:
    """가우시안 노이즈 추가 (std 는 [0,1] 스케일 기준)."""

    def __init__(self, p: float = 0.5, std_min: float = 0.01, std_max: float = 0.05):
        self.p = p
        self.std_min = std_min
        self.std_max = std_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(())) >= self.p:
            return x
        std = float(torch.empty(()).uniform_(self.std_min, self.std_max))
        return (x + torch.randn_like(x) * std).clamp(0, 1)


class BlackDotNoise:
    """검은색 점 형태의 작은 contamination (EM 흔한 artifact).

    n_dots 개의 작은 원형 점을 어두운 값으로 찍는다. radius 픽셀 단위.
    """

    def __init__(self, p: float = 0.5, n_min: int = 1, n_max: int = 12,
                 r_min: int = 1, r_max: int = 3, value: float = 0.0):
        self.p = p
        self.n_min, self.n_max = n_min, n_max
        self.r_min, self.r_max = r_min, r_max
        self.value = value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(())) >= self.p:
            return x
        _, h, w = x.shape
        n = int(torch.randint(self.n_min, self.n_max + 1, ()))
        yy = torch.arange(h).view(h, 1)
        xx = torch.arange(w).view(1, w)
        for _ in range(n):
            cy = int(torch.randint(0, h, ()))
            cx = int(torch.randint(0, w, ()))
            r = int(torch.randint(self.r_min, self.r_max + 1, ()))
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r  # (h,w)
            x[:, mask] = self.value
        return x


def build_em_eval_transform(image_size: int) -> v2.Compose:
    """clean eval transform (augmentation 없음). 학습 global crop 해상도에 맞춤."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            PerImageNormalize(),
        ]
    )


def build_em_train_transform(
    image_size: int,
    crop_scale: float = 0.8,
    gamma_p: float = 0.5,
    gamma_log_range: float = 0.4,
    noise_p: float = 0.5,
    noise_std_max: float = 0.05,
    blackdot_p: float = 0.5,
    rot90: bool = True,
    flip: bool = True,
) -> v2.Compose:
    """EM classification 학습용 augmentation transform.

    밝기(감마)/90도 회전/flip/노이즈/0.8배 크롭/검은점 — 사용자 지정 항목 반영.
    """
    steps = [v2.ToImage()]
    # 0.8배 수준 크롭 (scale 하한 crop_scale ~ 1.0 에서 RandomResizedCrop)
    steps.append(
        v2.RandomResizedCrop(
            image_size, scale=(crop_scale, 1.0), ratio=(0.9, 1.1),
            interpolation=v2.InterpolationMode.BICUBIC, antialias=True,
        )
    )
    if rot90:
        steps.append(RandomRot90(p=1.0))
    if flip:
        steps.append(v2.RandomHorizontalFlip(0.5))
        steps.append(v2.RandomVerticalFlip(0.5))
    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps.append(GammaJitter(p=gamma_p, log_range=gamma_log_range))
    steps.append(GaussianNoise(p=noise_p, std_max=noise_std_max))
    steps.append(BlackDotNoise(p=blackdot_p))
    steps.append(PerImageNormalize())
    return v2.Compose(steps)
