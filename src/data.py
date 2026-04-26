"""
NHATS CDT dataset: loading, participant-disjoint splits, class imbalance handling.

Rubric items:
  - #1  Train/val/test split (70/15/15 ratios)
  - #3  DataLoader batching + shuffling
  - #8  Normalization of input
  - #9  Basic preprocessing (resize, format conversion)
  - #10 Preprocessing pipeline addressing >=2 data quality challenges
        Challenge 1: Class imbalance (weighted sampling + class weights)
        Challenge 2: Image quality / format heterogeneity (TIFF -> RGB, resize)
        Challenge 3: Participant leakage across longitudinal rounds (disjoint split)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    DataConfig, SEED, NUM_CLASSES, LABELS_CSV, RAW_IMAGES_DIR,
)


class CDTDataset(Dataset):
    """
    Clock Drawing Test dataset.

    Expects a dataframe with columns:
      - image_path: .tif/.png/.jpg
      - label: integer 0–5
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # PIL convert("RGB") handles TIFF and standardises channel count
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
    Read the labels CSV and join with image filepaths.
    Drops rows whose image file does not exist on disk.
    """
    df = pd.read_csv(labels_csv)
    df = df.rename(columns={score_column: "label", id_column: "participant_id"})
    if "image_path" not in df.columns:
        df["image_path"] = df["participant_id"].apply(
            lambda pid: str(images_dir / f"{pid}{image_ext}")
        )
    # keep only valid score range
    df = df[df["label"].between(0, NUM_CLASSES - 1)].copy()
    df["label"] = df["label"].astype(int)
    # drop rows whose image is missing from disk
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    dropped = (~exists_mask).sum()
    if dropped > 0:
        print(f"[data] dropped {dropped} rows with missing image files")
    df = df[exists_mask].reset_index(drop=True)
    print(f"[data] loaded {len(df)} labeled clock images")
    print(f"[data] class distribution:\n{df['label'].value_counts().sort_index()}")
    return df


def stratified_split(df: pd.DataFrame, cfg: DataConfig):
    """Legacy random-stratified split. Kept for backward compatibility."""
    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6
    train_df, temp_df = train_test_split(
        df, test_size=1 - cfg.train_ratio, stratify=df["label"], random_state=SEED,
    )
    val_size_within_temp = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size_within_temp,
        stratify=temp_df["label"], random_state=SEED,
    )
    print(f"[split] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def participant_disjoint_split(df: pd.DataFrame, cfg: DataConfig):
    """
    Leak-safe split: every participant appears in EXACTLY ONE of train/val/test.

    NHATS is longitudinal — the same SPID draws clocks across multiple rounds.
    A random-stratified split would put the same person's Round-5 clock in
    training and their Round-11 clock in test, inflating held-out accuracy.
    GroupShuffleSplit on participant_id prevents that leakage.
    """
    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    # Step 1: train vs (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=cfg.train_ratio, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=df["participant_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df  = df.iloc[temp_idx].reset_index(drop=True)

    # Step 2: val vs test within the temp set
    val_ratio_in_temp = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio_in_temp, random_state=SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["participant_id"]))
    val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Sanity-check: assert zero participant overlap across splits
    train_ids = set(train_df["participant_id"])
    val_ids   = set(val_df["participant_id"])
    test_ids  = set(test_df["participant_id"])
    assert not (train_ids & val_ids),  "Participant leak: train/val overlap"
    assert not (train_ids & test_ids), "Participant leak: train/test overlap"
    assert not (val_ids   & test_ids), "Participant leak: val/test overlap"

    print(f"[split] participant-disjoint split:")
    print(f"  train: {len(train_df):,} clocks from {len(train_ids):,} participants")
    print(f"  val:   {len(val_df):,} clocks from {len(val_ids):,} participants")
    print(f"  test:  {len(test_df):,} clocks from {len(test_ids):,} participants")
    print(f"\n  train class dist:\n{train_df['label'].value_counts().sort_index().to_string()}")
    print(f"\n  test  class dist:\n{test_df['label'].value_counts().sort_index().to_string()}")
    return train_df, val_df, test_df


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    sklearn 'balanced' class weights: n_samples / (n_classes * count_y).
    Used in nn.CrossEntropyLoss(weight=...) so rare classes have a higher
    loss contribution. Computed only on the training split to avoid leakage.
    """
    labels = train_df["label"].values
    classes_present = np.unique(labels)
    weights_present = compute_class_weight(
        class_weight="balanced", classes=classes_present, y=labels,
    )
    # classes absent from train get weight 1.0
    full = np.ones(NUM_CLASSES, dtype=np.float32)
    for c, w in zip(classes_present, weights_present):
        full[c] = w
    return torch.tensor(full, dtype=torch.float32)


def make_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Oversample minority classes by drawing each example with
    probability proportional to 1 / class_count.
    Can be used instead of (or alongside) class-weighted loss.
    """
    class_counts = train_df["label"].value_counts().sort_index()
    sample_weights = train_df["label"].apply(lambda y: 1.0 / class_counts[y]).values
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )


def build_dataloaders(train_ds, val_ds, test_ds, cfg: DataConfig, sampler=None):
    """Wrap datasets in DataLoaders with batching, shuffling, and pin_memory."""
    shuffle_train = sampler is None  # can't both shuffle=True and pass a sampler
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=shuffle_train,
        sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
