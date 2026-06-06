# Custom dataset for loading images from a pandas DataFrame.
# Supports various image formats (jpg, tif, png, etc.)
# and automatically converts grayscale images to 3-channel RGB.
# Optionally loads pre-computed denoised target from 'filtered_filepath' column.

import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a DataFrame from CSV, Pickle, or Parquet (auto-detected by extension)."""
    p = path.lower()
    if p.endswith('.csv'):
        return pd.read_csv(path)
    elif p.endswith('.pkl') or p.endswith('.pickle'):
        return pd.read_pickle(path)
    elif p.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}  (use .csv, .pkl, .pickle, or .parquet)")


def save_dataframe(df: pd.DataFrame, path: str):
    """Save a DataFrame to CSV, Pickle, or Parquet (auto-detected by extension)."""
    p = path.lower()
    if p.endswith('.csv'):
        df.to_csv(path, index=False)
    elif p.endswith('.pkl') or p.endswith('.pickle'):
        df.to_pickle(path)
    elif p.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path}  (use .csv, .pkl, .pickle, or .parquet)")


class PerImageNormalize:
    """Per-image (instance) normalization.

    For each image tensor independently, computes its own mean and std
    and normalizes to zero-mean / unit-variance.  This removes
    domain-specific brightness/contrast differences (e.g. TEM vs SEM).
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: (C, H, W)
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True)
        return (tensor - mean) / (std + 1e-6)


class CSVImageDataset(Dataset):
    """Dataset that reads image file paths from a pandas DataFrame.

    Args:
        df: pandas DataFrame containing file paths.
        filepath_col: column name for the original image file paths.
        filtered_col: column name for pre-computed denoised image paths.
            If the column exists and use_filtered_target=True, __getitem__
            returns (input_tensor, filtered_tensor) for dual-target loss.
        use_filtered_target: whether to load pre-computed filtered targets.
        transform: torchvision transforms to apply (shared for both images).
    """

    def __init__(self, df, filepath_col='filepath',
                 filtered_col='filtered_filepath',
                 use_filtered_target=False, transform=None):
        self.paths = df[filepath_col].tolist()
        self.transform = transform
        self.use_filtered_target = use_filtered_target

        if use_filtered_target:
            if filtered_col not in df.columns:
                raise ValueError(
                    f"Column '{filtered_col}' not found in DataFrame. "
                    f"Run precompute_denoise.py first to create filtered images.")
            self.filtered_paths = df[filtered_col].tolist()
        else:
            self.filtered_paths = None

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _load_rgb(path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        img = self._load_rgb(self.paths[idx])

        if self.use_filtered_target and self.filtered_paths is not None:
            filtered_img = self._load_rgb(self.filtered_paths[idx])

            # Apply identical random transforms via seed synchronization
            seed = random.randint(0, 2**31)

            random.seed(seed)
            torch.manual_seed(seed)
            img_tensor = self.transform(img)

            random.seed(seed)
            torch.manual_seed(seed)
            target_tensor = self.transform(filtered_img)

            return img_tensor, target_tensor

        # Default: no filtered target
        if self.transform is not None:
            img = self.transform(img)

        # Return dummy label 0 for compatibility with engine_pretrain
        return img, 0
