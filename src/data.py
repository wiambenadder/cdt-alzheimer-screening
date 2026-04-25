"""
We have NHATS CDT dataset loader with train/val/test splits, class imbalance handling,
and PyTorch DataLoader construction.

This helps us addresse 2 data quality challenges:
  1. Class imbalance: weighted sampling + class weights
  2. Image heterogeneity: from TIFF to RGB conversion, normalization, resizing
"""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    DataConfig, SEED, NUM_CLASSES, LABELS_CSV, RAW_IMAGES_DIR,
)


class CDTDataset(Dataset):
    """
    This is for the Clock Drawing Test dataset.

    A dataframe with columns:
      - image_path: .tif/.png/.jpg
      - label: integer from 0 to 5 
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # PIL handles TIFF by converting to RGB to standardize channel count
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])



def load_labels(
    labels_csv: Path = LABELS_CSV,
    images_dir: Path = RAW_IMAGES_DIR,
    image_ext: str = ".tif",
    score_column: str = "cdt_score",
    id_column: str = "participant_id",
) -> pd.DataFrame:
    """
    This reads the labels on the CSV and joins them with the image filepaths.
    """
    df = pd.read_csv(labels_csv)
    df = df.rename(columns={score_column: "label", id_column: "participant_id"})
    df["image_path"] = df["participant_id"].apply(
        lambda pid: str(images_dir / f"{pid}{image_ext}")
    )
    # only use scores from 0 to 5
    df = df[df["label"].between(0, NUM_CLASSES - 1)].copy()
    df["label"] = df["label"].astype(int)
    # we only keep rows whose image actually exists on the disk
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    dropped = (~exists_mask).sum()
    if dropped > 0:
        print(f"[data] dropped {dropped} rows with missing image files")
    df = df[exists_mask].reset_index(drop=True)
    print(f"[data] loaded {len(df)} labeled clock images")
    print(f"[data] class distribution:\n{df['label'].value_counts().sort_index()}")
    return df


def stratified_split(df: pd.DataFrame, cfg: DataConfig):
    """
    70/15/15 train/val/test split on the CDT label.
    """
    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df,
        test_size=1 - cfg.train_ratio,
        stratify=df["label"],
        random_state=SEED,
    )
    # temp_df is the combined val+test
    val_size_within_temp = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size_within_temp,
        stratify=temp_df["label"],
        random_state=SEED,
    )
    print(f"[split] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    sklearn 'balanced' class weights: n_samples / (n_classes * count_y).
    Used in nn.CrossEntropyLoss(weight=...) so rare classes have higher loss 
    contribution
    Computed only on the training split to avoid any test set leakage.
    """
    labels = train_df["label"].values
    classes_present = np.unique(labels)
    weights_present = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=labels,
    )
    # classes not seen in train get weight 1.0
    full = np.ones(NUM_CLASSES, dtype=np.float32)
    for c, w in zip(classes_present, weights_present):
        full[c] = w
    return torch.tensor(full, dtype=torch.float32)


def make_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Instead of class weighting: this oversamples the minority classes by
    drawing each example with probability ~ 1/class_count.
    """
    class_counts = train_df["label"].value_counts().sort_index()
    sample_weights = train_df["label"].apply(lambda y: 1.0 / class_counts[y]).values
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(train_ds, val_ds, test_ds, cfg: DataConfig, sampler=None):
    """Wrap datasets in DataLoaders with batching, shuffling, pin_memory."""
    shuffle_train = sampler is None  # can't both shuffle and use sampler
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
