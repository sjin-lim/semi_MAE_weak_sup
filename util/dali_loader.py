"""
NVIDIA DALI-based DataLoader for MAE pretraining.

Replaces the CPU-bound PyTorch DataLoader with a GPU-accelerated pipeline:
  - Raw compressed bytes (JPEG/PNG/TIFF) are read from disk on CPU
  - Image decoding (nvJPEG for JPEG) and all augmentations run on GPU
  - Eliminates decoded float32 image tensors from system RAM entirely

Usage in main_pretrain.py:
    from util.dali_loader import build_dali_loader
    data_loader_train = build_dali_loader(df, ...)
"""

import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


# ---------------------------------------------------------------------------
# ExternalCSVSource: per-sample callable consumed by fn.external_source
# ---------------------------------------------------------------------------

class ExternalCSVSource:
    """
    Provides raw image bytes to a DALI pipeline from a pandas DataFrame.

    Handles:
    - Multi-GPU sharding (replaces DistributedSampler)
    - Epoch-level shuffling (call set_epoch before each epoch)
    - Optional dual-target mode (original + filtered image pairs)

    DALI calls __call__(sample_info) once per sample. sample_info.idx_in_epoch
    is the position within the current epoch's shuffled index list.
    StopIteration signals end-of-epoch to DALI.
    """

    def __init__(self, df, filepath_col, batch_size,
                 shard_id, num_shards,
                 shuffle=True, seed=0,
                 filtered_col=None, use_filtered_target=False):
        self.paths = df[filepath_col].tolist()
        self.batch_size = batch_size
        self.use_filtered_target = use_filtered_target
        self.filtered_paths = (
            df[filtered_col].tolist()
            if use_filtered_target and filtered_col and filtered_col in df.columns
            else None
        )

        # Sharding: pad total length to multiple of (num_shards * batch_size),
        # then assign a contiguous slice to this rank.
        n = len(self.paths)
        pad = (
            (num_shards * batch_size - n % (num_shards * batch_size))
            % (num_shards * batch_size)
        )
        indices = list(range(n)) + list(range(pad))   # wrap-around padding
        shard_size = len(indices) // num_shards
        self._shard_indices = np.array(
            indices[shard_id * shard_size : (shard_id + 1) * shard_size],
            dtype=np.int64,
        )

        self._shuffle = shuffle
        self._seed = seed
        self._epoch_indices = self._shard_indices.copy()

    def __len__(self):
        """Number of batches per epoch for this shard."""
        return len(self._shard_indices) // self.batch_size

    def set_epoch(self, epoch: int):
        """Reshuffle indices for the given epoch. Call before each epoch."""
        self._epoch_indices = self._shard_indices.copy()
        if self._shuffle:
            rng = np.random.default_rng(self._seed + epoch)
            rng.shuffle(self._epoch_indices)

    def __call__(self, sample_info):
        """
        Called by DALI once per sample.

        sample_info.idx_in_epoch: position within this epoch (0-based)
        sample_info.iteration:    batch index within this epoch

        Returns raw bytes as np.ndarray(dtype=uint8).
        For dual-target mode, returns a tuple (raw, raw_filtered).
        """
        # Signal end-of-epoch when all batches have been consumed.
        if sample_info.iteration >= len(self._shard_indices) // self.batch_size:
            raise StopIteration

        idx = int(self._epoch_indices[
            sample_info.idx_in_epoch % len(self._epoch_indices)
        ])

        with open(self.paths[idx], 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)

        if self.use_filtered_target and self.filtered_paths:
            with open(self.filtered_paths[idx], 'rb') as f:
                raw_filtered = np.frombuffer(f.read(), dtype=np.uint8)
            return raw, raw_filtered

        return raw


# ---------------------------------------------------------------------------
# DALI pipelines
# ---------------------------------------------------------------------------

def _normalize_single(img, instance_norm, mean, std):
    """Apply normalization to a single HWC float image in [0, 1]."""
    if instance_norm:
        # Per-image channel-wise normalization (matches PerImageNormalize)
        # img shape: [H, W, C]; reduce over spatial axes [0, 1]
        m = fn.reductions.mean(img, axes=[0, 1], keep_dims=True)
        s = fn.reductions.std_dev(img, axes=[0, 1], keep_dims=True, ddof=1)
        img = (img - m) / fn.math.max(s, 1e-6)
    else:
        # Standard ImageNet normalization
        # fn.normalize expects mean/std as lists matching channel count
        img = fn.normalize(img, device='gpu', mean=mean, stddev=std)
    return img


@pipeline_def
def _mae_pipeline_single(source, input_size, instance_norm, mean, std):
    """
    Single-target DALI pipeline.

    Steps: ExternalSource -> GPU decode -> RandomResizedCrop ->
           RandomHFlip -> /255 -> Normalize -> HWC→CHW
    """
    raw = fn.external_source(
        source=source,
        dtype=types.UINT8,
        batch=False,        # per-sample callback mode
    )

    # GPU decode via nvJPEG (JPEG) or CPU fallback (PNG/TIFF) + GPU transfer
    img = fn.decoders.image(raw, device='mixed', output_type=types.RGB)

    # RandomResizedCrop: matches torchvision RandomResizedCrop(size, scale=(0.2,1.0))
    img = fn.random_resized_crop(
        img,
        device='gpu',
        size=input_size,
        random_area=(0.2, 1.0),          # same semantics as torchvision scale=
        random_aspect_ratio=(0.75, 1.333),  # torchvision default ratio
        interp_type=types.INTERP_CUBIC,   # bicubic = interpolation=3
    )

    # RandomHorizontalFlip
    coin = fn.random.coin_flip(probability=0.5)
    img = fn.flip(img, device='gpu', horizontal=coin)

    # Scale to [0, 1]
    img = fn.cast(img, dtype=types.FLOAT, device='gpu')
    img = img / 255.0

    # Normalization
    img = _normalize_single(img, instance_norm, mean, std)

    # HWC -> CHW (PyTorch convention)
    img = fn.transpose(img, device='gpu', perm=[2, 0, 1])

    return img


@pipeline_def
def _mae_pipeline_dual(source, input_size, instance_norm, mean, std):
    """
    Dual-target DALI pipeline for edge-preserving denoise loss.

    Applies identical spatial augmentations to both original and filtered
    images using the 6-channel concat trick: concatenate [H,W,3]+[H,W,3]
    -> [H,W,6], apply one random_resized_crop + flip, then split back.
    This guarantees pixel-perfect alignment between both outputs.
    """
    raw, raw_filtered = fn.external_source(
        source=source,
        num_outputs=2,
        dtype=types.UINT8,
        batch=False,
    )

    img = fn.decoders.image(raw,          device='mixed', output_type=types.RGB)
    flt = fn.decoders.image(raw_filtered, device='mixed', output_type=types.RGB)

    # Concatenate along channel axis: [H, W, 3] + [H, W, 3] -> [H, W, 6]
    # fn.cat defaults to last axis for HWC layout
    combined = fn.cat(img, flt, axis=2)

    # One set of spatial transforms applied to the 6-channel combined tensor
    # fn.random_resized_crop is channel-independent -> same crop for both halves
    combined = fn.random_resized_crop(
        combined,
        device='gpu',
        size=input_size,
        random_area=(0.2, 1.0),
        random_aspect_ratio=(0.75, 1.333),
        interp_type=types.INTERP_CUBIC,
    )

    coin = fn.random.coin_flip(probability=0.5)
    combined = fn.flip(combined, device='gpu', horizontal=coin)

    combined = fn.cast(combined, dtype=types.FLOAT, device='gpu')
    combined = combined / 255.0

    # Split back into original and filtered
    img = combined[:, :, 0:3]
    flt = combined[:, :, 3:6]

    # Normalize each independently
    img = _normalize_single(img, instance_norm, mean, std)
    flt = _normalize_single(flt, instance_norm, mean, std)

    # HWC -> CHW
    img = fn.transpose(img, device='gpu', perm=[2, 0, 1])
    flt = fn.transpose(flt, device='gpu', perm=[2, 0, 1])

    return img, flt


# ---------------------------------------------------------------------------
# DALILoaderWrapper: transparent (samples, targets) interface
# ---------------------------------------------------------------------------

class DALILoaderWrapper:
    """
    Wraps DALIGenericIterator to yield (samples, targets) tuples,
    matching the interface expected by engine_pretrain.train_one_epoch.

    DALIGenericIterator yields: [{'samples': Tensor, ...}]  (list of 1 dict)
    This wrapper yields:        (samples_Tensor, targets_Tensor_or_0)

    Also exposes _source for set_epoch() and __len__ for LR scheduling.
    """

    def __init__(self, dali_iter, source, use_filtered_target=False):
        self._iter = dali_iter
        self._source = source
        self._use_filtered = use_filtered_target

    def __len__(self):
        return len(self._source)   # shard_size // batch_size

    def __iter__(self):
        for batch in self._iter:
            data = batch[0]        # first (and only) pipeline output dict
            samples = data['samples']
            targets = data['targets'] if self._use_filtered else 0
            yield samples, targets


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_dali_loader(
    df,
    filepath_col,
    filtered_col,
    use_filtered_target,
    instance_norm,
    input_size,
    batch_size,
    num_threads,
    device_id,
    shard_id,
    num_shards,
    seed=0,
):
    """
    Build and return a DALILoaderWrapper for MAE pretraining.

    Args:
        df:                   pandas DataFrame with image paths
        filepath_col:         column name for original image paths
        filtered_col:         column name for pre-computed filtered image paths
        use_filtered_target:  enable dual-target (denoise) loss mode
        instance_norm:        use per-image normalization instead of ImageNet stats
        input_size:           spatial crop size (square, e.g. 448)
        batch_size:           batch size per GPU
        num_threads:          DALI internal CPU thread count for file I/O
        device_id:            GPU index for this process (local_rank)
        shard_id:             data shard index for this process (global_rank)
        num_shards:           total number of data shards (world_size)
        seed:                 base random seed

    Returns:
        DALILoaderWrapper that yields (samples, targets) tuples.
        Tensors are already on the correct GPU device.
    """
    if not DALI_AVAILABLE:
        raise ImportError(
            "nvidia-dali-cuda120 (or matching cuda variant) is not installed. "
            "Install with: pip install nvidia-dali-cuda120"
        )

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    source = ExternalCSVSource(
        df=df,
        filepath_col=filepath_col,
        batch_size=batch_size,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle=True,
        seed=seed,
        filtered_col=filtered_col,
        use_filtered_target=use_filtered_target,
    )

    pipeline_fn = _mae_pipeline_dual if use_filtered_target else _mae_pipeline_single

    pipe = pipeline_fn(
        source=source,
        input_size=input_size,
        instance_norm=instance_norm,
        mean=mean,
        std=std,
        # @pipeline_def keyword args
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed + device_id,   # per-device seed avoids identical augmentations
    )
    pipe.build()

    output_map = ['samples', 'targets'] if use_filtered_target else ['samples']

    dali_iter = DALIGenericIterator(
        pipe,
        output_map=output_map,
        size=len(source) * batch_size,   # total samples per epoch for this shard
        auto_reset=True,                 # reset pipeline at end of epoch
        last_batch_policy=LastBatchPolicy.DROP,  # matches drop_last=True
    )

    return DALILoaderWrapper(dali_iter, source, use_filtered_target)
