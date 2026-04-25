"""
Training loop.

Rubric items this file supports:
  - #2   Training curves (TensorBoard scalars for loss + acc)
  - #5   Regularization: weight_decay (L2) + dropout + early stopping
  - #14  Learning rate scheduling (cosine / plateau)
  - #16  Gradient clipping, mixed precision training
  - #17  GPU/CUDA acceleration
  - #19  Optimizer comparison (switch via TrainConfig.optimizer)
"""
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TrainConfig, MODELS_DIR, RESULTS_DIR
from .models import get_model
from .utils import set_seed, save_checkpoint, get_device, count_parameters


# -----------------------------------------------------------------------------
# Optimizers and schedulers
# -----------------------------------------------------------------------------
def build_optimizer(backbone_params, head_params, cfg: TrainConfig):
    """
    Differential learning rates: lower for pretrained backbone, higher
    for newly initialized head. Standard transfer-learning practice.
    """
    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": cfg.lr_head})

    if cfg.optimizer == "adamw":
        return AdamW(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return Adam(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    if cfg.lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs * steps_per_epoch)
    if cfg.lr_scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    if cfg.lr_scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler {cfg.lr_scheduler!r}")


# -----------------------------------------------------------------------------
# Per-epoch train / eval
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, cfg: TrainConfig, epoch: int, writer=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"epoch {epoch} [train]", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=cfg.mixed_precision):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale before clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        # Step per-batch scheduler (cosine); plateau steps on val loss later
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.3f}")

    tr_loss = running_loss / total
    tr_acc = correct / total
    if writer is not None:
        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("train/acc", tr_acc, epoch)
        for i, pg in enumerate(optimizer.param_groups):
            writer.add_scalar(f"train/lr_group_{i}", pg["lr"], epoch)
    return tr_loss, tr_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             epoch: Optional[int] = None, writer=None, tag: str = "val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    ev_loss = running_loss / total
    ev_acc = correct / total
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{tag}/loss", ev_loss, epoch)
        writer.add_scalar(f"{tag}/acc", ev_acc, epoch)
    return ev_loss, ev_acc


# -----------------------------------------------------------------------------
# Top-level train() entry point
# -----------------------------------------------------------------------------
def train(model_name: str,
          train_loader, val_loader,
          class_weights: torch.Tensor,
          cfg: TrainConfig,
          run_name: str,
          freeze_backbone: bool = False) -> dict:
    """
    Train one model and return a dict summary:
      {run_name, history, best_val_loss, best_epoch, checkpoint_path}
    """
    set_seed()
    device = get_device()
    print(f"[train] run={run_name} device={device}")

    model, backbone_params, head_params = get_model(
        model_name,
        freeze_backbone=freeze_backbone,
        dropout=cfg.dropout,
    )
    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"[train] params total={total:,} trainable={trainable:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = build_optimizer(backbone_params, head_params, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=(cfg.mixed_precision and device == "cuda"))

    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb")

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    history = []
   # to always save to Drive, even if cfg.MODELS_DIR override didn't propagate
   import os
   _drive_models = "/content/drive/MyDrive/cdt-data/models"
   if os.path.exists("/content/drive/MyDrive/cdt-data"):
       os.makedirs(_drive_models, exist_ok=True)
       ckpt_path = type(MODELS_DIR)(_drive_models) / f"{run_name}_best.pt"
   else:
       ckpt_path = MODELS_DIR / f"{run_name}_best.pt"

    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, cfg, epoch, writer,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, writer, tag="val",
        )
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"[train] ep={epoch:02d} tr_loss={tr_loss:.4f} "
              f"tr_acc={tr_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            save_checkpoint(model, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"[train] early stop at epoch {epoch} (no val improvement for "
                      f"{cfg.early_stopping_patience} epochs)")
                break

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    writer.close()

    return {
        "run_name": run_name,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
    }
