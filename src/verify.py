"""
In this file we have the smoke test: it confirms the full pipeline runs end-to-end without training.

We use:  python -m src.verify

Exits with code 0 if everything is wired correctly, which means our pipeline was successful.
"""
import sys
import torch

from .config import DataConfig, AugConfig, LABELS_CSV, RAW_IMAGES_DIR, NUM_CLASSES
from .data import load_labels, stratified_split, compute_class_weights, \
                  CDTDataset, build_dataloaders
from .augmentation import build_train_transform, build_eval_transform
from .models import get_model
from .utils import get_device, count_parameters


def main():
    print("=" * 60)
    print("CDT Alzheimer Screening - pipeline smoke test")
    print("=" * 60)

    # 1. the data
    if not LABELS_CSV.exists():
        print(f"[fail] labels CSV not found at {LABELS_CSV}")
        print("       see data/README.md for how to set up NHATS data")
        sys.exit(1)
    df = load_labels()
    if len(df) == 0:
        print("[fail] no valid rows in labels CSV - check image paths")
        sys.exit(1)

    data_cfg = DataConfig(batch_size=4, num_workers=0)
    train_df, val_df, test_df = stratified_split(df, data_cfg)
    class_weights = compute_class_weights(train_df)
    print(f"[ok] class weights: {class_weights.tolist()}")

    # 2. transforms
    aug_cfg = AugConfig()
    train_tf = build_train_transform(data_cfg.image_size, aug_cfg, use_aug=True)
    eval_tf = build_eval_transform(data_cfg.image_size)

    train_ds = CDTDataset(train_df.head(8), transform=train_tf)
    val_ds   = CDTDataset(val_df.head(4),   transform=eval_tf)
    test_ds  = CDTDataset(test_df.head(4),  transform=eval_tf)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds, data_cfg,
    )

    # 3. the models: the forward pass on one batch each
    device = get_device()
    print(f"[ok] device: {device}")
    imgs, labels = next(iter(train_loader))
    print(f"[ok] sample batch shape: {imgs.shape}, labels: {labels.tolist()}")

    for name in ["vgg16", "efficientnet_b0", "vit_b16"]:
        model, bb, head = get_model(name, freeze_backbone=False, dropout=0.3)
        model = model.to(device)
        total, trainable = count_parameters(model)
        with torch.no_grad():
            logits = model(imgs.to(device))
        assert logits.shape == (imgs.size(0), NUM_CLASSES), \
            f"{name}: expected ({imgs.size(0)},{NUM_CLASSES}), got {logits.shape}"
        print(f"[ok] {name}: forward pass ok, params total={total:,} trainable={trainable:,}")

    print("=" * 60)
    print("[ok] all checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
