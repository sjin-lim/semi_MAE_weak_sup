# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler

from .datasets import ADE20K, CocoCaptions, ImageNet, ImageNet22k, NYU, SemiMAEDataset, SemiMAEWebDataset
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler

logger = logging.getLogger("dinov3")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(
    image_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    # SemiMAE: h5_path=/path/to/file.h5 또는 csv_path=... 또는 root=...
    # 지원 키: h5_path, csv_path, root, filepath_col,
    #           use_filtered_target, filtered_col
    if name == "SemiMAE":
        SEMI_MAE_KEYS = {
            "h5_path", "csv_path", "root", "filepath_col",
            "use_filtered_target", "filtered_col", "preload",
        }
        SEMI_MAE_BOOL_KEYS = {"use_filtered_target", "preload"}
        for token in tokens[1:]:
            key, value = token.split("=", 1)
            assert key in SEMI_MAE_KEYS, f"Unknown SemiMAE key: {key}"
            if key in SEMI_MAE_BOOL_KEYS:
                value = value.lower() in ("1", "true", "yes")
            kwargs[key] = value
        return SemiMAEDataset, kwargs

    if name == "WebDataset":
        WDS_KEYS = {"path", "buffer_size", "shardshuffle", "min_std", "pipeline_restart_interval"}
        WDS_INT_KEYS = {"buffer_size", "pipeline_restart_interval"}
        WDS_FLOAT_KEYS = {"min_std"}
        WDS_BOOL_KEYS = {"shardshuffle"}
        for token in tokens[1:]:
            key, value = token.split("=", 1)
            assert key in WDS_KEYS, f"Unknown WebDataset key: {key}"
            if key in WDS_INT_KEYS:
                value = int(value)
            elif key in WDS_FLOAT_KEYS:
                value = float(value)
            elif key in WDS_BOOL_KEYS:
                value = value.lower() in ("1", "true", "yes")
            kwargs[key] = value
        return SemiMAEWebDataset, kwargs

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    elif name == "ADE20K":
        class_ = ADE20K
        if "split" in kwargs:
            kwargs["split"] = ADE20K.Split[kwargs["split"]]
    elif name == "CocoCaptions":
        class_ = CocoCaptions
        if "split" in kwargs:
            kwargs["split"] = CocoCaptions.Split[kwargs["split"]]
    elif name == "NYU":
        class_ = NYU
        if "split" in kwargs:
            kwargs["split"] = NYU.Split[kwargs["split"]]
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.
        transforms: A transform to apply to both images and targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)

    # SemiMAEDataset / SemiMAEWebDataset: transforms= 인수로 DataAugmentationDINO 전달.
    # train.py 는 transform= 으로 넘기므로 fallback 처리.
    if class_ in (SemiMAEDataset, SemiMAEWebDataset):
        effective_transforms = transforms if transforms is not None else transform
        dataset = class_(transforms=effective_transforms, **kwargs)
    else:
        dataset = class_(transform=transform, target_transform=target_transform, transforms=transforms, **kwargs)

    if isinstance(dataset, torch.utils.data.IterableDataset):
        logger.info(f"# of shards: {len(dataset.urls):,d}  (IterableDataset — len() 불가)")
    else:
        logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        dataset.transform = transform
    if not hasattr(dataset, "target_transform"):
        dataset.target_transform = target_transform
    if not hasattr(dataset, "transforms"):
        dataset.transforms = transforms

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    worker_init_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
        worker_init_fn: Optional init function for each dataloader worker.
    """

    # IterableDataset (WebDataset 등): sampler 불가, 셔플/분배는 dataset 내부에서 처리
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)

    if is_iterable:
        sampler = None
        effective_persistent = False
    else:
        sampler = _make_sampler(
            dataset=dataset,
            type=sampler_type,
            shuffle=shuffle,
            seed=seed,
            size=sampler_size,
            advance=sampler_advance,
        )
        # num_workers > 0 인 경우 persistent_workers 를 자동으로 활성화.
        effective_persistent = persistent_workers or (num_workers > 0)

    # worker 프로세스는 메인의 gc.disable() 을 상속받아 GC가 꺼진 상태.
    # worker에서 GC를 재활성화 + malloc_trim 콜백 등록으로:
    # 1. PIL 순환참조를 worker가 자체 수거
    # 2. numpy/PIL/cv2의 반복적 alloc/free로 인한 힙 단편화를 정리
    if num_workers > 0:
        import gc as _gc

        def _enable_gc_in_worker(worker_id):
            _gc.enable()
            # GC 콜백으로 malloc_trim 주기적 실행 — worker의 힙 단편화 방지
            try:
                import ctypes as _ctypes
                _libc = _ctypes.CDLL("libc.so.6")
                def _trim_callback(phase, info):
                    if phase == "stop":  # GC 완료 후 실행
                        _libc.malloc_trim(0)
                _gc.callbacks.append(_trim_callback)
            except OSError:
                pass  # non-glibc 환경

        if worker_init_fn is None:
            worker_init_fn = _enable_gc_in_worker
        else:
            _orig_init = worker_init_fn
            def _chained_init(worker_id):
                _enable_gc_in_worker(worker_id)
                _orig_init(worker_id)
            worker_init_fn = _chained_init

    logger.info("using PyTorch data loader")

    # prefetch_factor: worker가 미리 준비해둘 배치 수.
    # resampled=True + split_by_node WebDataset에서는 deadlock 위험 없음.
    # GPU idle을 줄여 throughput 향상.
    prefetch = 2 if num_workers > 0 else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=effective_persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
