"""
Training loop.

What does this file do:
  1. Training curves (TensorBoard scalars for loss, accuracy, kappa and LR)
  2. Regularization: weight_decay (L2) + dropout + early stopping
  3. Learning rate scheduling (cosine / plateau)
  4. Gradient clipping, mixed precision training on CUDA
  5. The optimizer switch using TrainConfig.optimizer

Checkpoints go to cfg.MODELS_DIR. We read it when train() runs, so setting
cfg.MODELS_DIR in a notebook (or CDT_MODELS_DIR) works.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from . import config as cfg_module
from .config import TrainConfig, SEED
from .models import get_model
from .utils import set_seed, save_checkpoint, get_device, count_parameters


# These are optimizers and schedulers
def build_optimizer(backbone_params, head_params, cfg: TrainConfig):
    """
    This used differential learning rates: it is lower for pretrained backbone and higher
    for newly initialized head. This is how it is done in transfer learning
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


# per-epoch training / evaluation
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, cfg: TrainConfig, epoch: int, writer=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    use_amp = cfg.mixed_precision and device == "cuda"
    pbar = tqdm(loader, desc=f"epoch {epoch} [train]", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # we unscale before clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        # per-batch scheduler (cosine) and plateau steps on val loss later
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
    """We return (loss, accuracy, quadratic kappa) on a loader."""
    model.eval()
    running_loss, total = 0.0, 0
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        total += labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    ev_loss = running_loss / total
    ev_acc = float((y_true == y_pred).mean())
    # just to check, kappa is undefined with a single class
    if len(np.unique(np.concatenate([y_true, y_pred]))) > 1:
        ev_kappa = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    else:
        ev_kappa = 0.0

    if writer is not None and epoch is not None:
        writer.add_scalar(f"{tag}/loss", ev_loss, epoch)
        writer.add_scalar(f"{tag}/acc", ev_acc, epoch)
        writer.add_scalar(f"{tag}/kappa", ev_kappa, epoch)
    return ev_loss, ev_acc, ev_kappa


def train(model_name: str,
          train_loader, val_loader,
          class_weights: torch.Tensor,
          cfg: TrainConfig,
          run_name: str,
          freeze_backbone: bool = False,
          seed: int = SEED,
          pretrained: bool = True) -> dict:
    """
    We train one model and return a dict summary:
      {run_name, seed, history, best_val_loss, best_val_kappa, best_epoch, checkpoint_path}
    """
    set_seed(seed)
    device = get_device()
    print(f"[train] run={run_name} seed={seed} device={device}")

    model, backbone_params, head_params = get_model(
        model_name,
        freeze_backbone=freeze_backbone,
        dropout=cfg.dropout,
        pretrained=pretrained,
    )
    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"[train] params total={total:,} trainable={trainable:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = build_optimizer(backbone_params, head_params, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision and device == "cuda"))

    run_dir = cfg_module.RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tb")
    ckpt_path = Path(cfg_module.MODELS_DIR) / f"{run_name}_best.pt"
    print(f"[train] checkpoints -> {ckpt_path}")

    if cfg.select_metric not in ("val_loss", "val_kappa"):
        raise ValueError(f"Unknown select_metric {cfg.select_metric!r}")

    best_val_loss = float("inf")
    best_val_kappa = -float("inf")
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, cfg, epoch, writer,
        )
        val_loss, val_acc, val_kappa = evaluate(
            model, val_loader, criterion, device, epoch, writer, tag="val",
        )
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_kappa": val_kappa,
        })
        print(f"[train] ep={epoch:02d} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_kappa={val_kappa:.3f}")

        # early stopping on whichever validation metric we chose
        if cfg.select_metric == "val_loss":
            improved = val_loss < best_val_loss
        else:
            improved = val_kappa > best_val_kappa

        if improved:
            best_val_loss = val_loss
            best_val_kappa = val_kappa
            best_epoch = epoch
            patience = 0
            save_checkpoint(model, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"[train] early stop at epoch {epoch} (no {cfg.select_metric} "
                      f"improvement for {cfg.early_stopping_patience} epochs)")
                break

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    writer.close()

    return {
        "run_name": run_name,
        "seed": seed,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_kappa": best_val_kappa,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
    }
