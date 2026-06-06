# Mixed batch loader: unlabeled (WebDataset) + labeled (folder Dataset).
#
# Strategy: 두 dataloader 를 병렬 운영하고 ratio 기반으로 batch 단위 mixing.
# 학습 코드는 batch 단위로 처리 (labeled 인지 unlabeled 인지는 batch flag 로 구분).

import logging
import random
from typing import Any, Dict, Iterator

logger = logging.getLogger("dinov3")


class RatioMixedLoader:
    """두 dataloader 를 ratio 기반으로 mixing.

    매 iteration 마다 random number 로 어느 loader 에서 가져올지 결정.
    Batch 형식이 다르므로 batch 에 'is_labeled' flag 를 추가하여 downstream 에서
    분기 처리.

    Args:
        unlabeled_loader: 기존 SSL WebDataset 기반 loader.
        labeled_loader:   LabeledEMDataset 기반 loader.
        labeled_ratio:    매 step 마다 labeled batch 사용할 확률 (0.0 ~ 1.0).
                          0.25 면 약 4 step 중 1 step 이 labeled.
        seed:             ratio 분기용 난수 seed.
    """

    def __init__(
        self,
        unlabeled_loader,
        labeled_loader,
        labeled_ratio: float = 0.25,
        seed: int = 0,
    ):
        self.unlabeled_loader = unlabeled_loader
        self.labeled_loader = labeled_loader
        self.labeled_ratio = max(0.0, min(1.0, labeled_ratio))
        self.rng = random.Random(seed)

        logger.info(
            f"[RatioMixedLoader] labeled_ratio={self.labeled_ratio}, seed={seed}"
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        unl_iter = iter(self.unlabeled_loader)
        lab_iter = iter(self.labeled_loader) if self.labeled_loader is not None else None

        while True:
            use_labeled = (
                lab_iter is not None
                and self.rng.random() < self.labeled_ratio
            )

            try:
                if use_labeled:
                    batch = next(lab_iter)
                    batch["is_labeled"] = True
                else:
                    batch = next(unl_iter)
                    batch["is_labeled"] = False
            except StopIteration:
                if use_labeled:
                    lab_iter = iter(self.labeled_loader)
                    batch = next(lab_iter)
                    batch["is_labeled"] = True
                else:
                    unl_iter = iter(self.unlabeled_loader)
                    batch = next(unl_iter)
                    batch["is_labeled"] = False

            yield batch
