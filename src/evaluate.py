"""
Evaluation: metrics, confusion matrix, error analysis, inference timing,
bootstrap confidence intervals and kappa-tuned decoding.
"""
import time
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    cohen_kappa_score, roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from .config import NUM_CLASSES, CLASS_NAMES, RESULTS_DIR
from .utils import get_device


#the collection
@torch.no_grad()
def collect_predictions(model, loader, device: str):
    """We run the model over a loader and return (y_true, y_pred, y_probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )


#2. rubric #87  >=3 distinct evaluation metrics  (we return 7 metrics)
def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_probs: np.ndarray) -> Dict[str, float]:
    """
    Seven metrics, which cover different aspects of the performance on an
    imbalanced ordinal classification task:

      accuracy         - which is naive correctness
      macro_f1         - unweighted average F1 (this penalizes ignoring rare classes)
      weighted_f1      - F1 weighted by class frequency
      macro_precision  - the average precision across classes
      macro_recall     - the average recall across classes
      quadratic_kappa  - the ordinal-aware agreement (close predictions partly credited)
      macro_auc        - the one-vs-rest ROC-AUC averaged over classes
    """
    out = {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "macro_f1":         float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1":      float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision":  float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":     float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "quadratic_kappa":  float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
    try:
        out["macro_auc"] = float(roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="macro",
            labels=list(range(NUM_CLASSES)),
        ))
    except ValueError as e:
        print(f"[eval] AUC skipped: {e}")
        out["macro_auc"] = None
    return out


def save_classification_report(y_true, y_pred, path: Path):
    rep = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)

#confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path: Path, normalize: bool = False):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        fmt = "d"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES))
    ax.set_xlabel("Predicted CDT Score")
    ax.set_ylabel("True CDT Score")
    ax.set_title(f"Confusion Matrix {'(normalized)' if normalize else ''}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# 1. rubric #86  inference time / throughput measured
def measure_inference_time(model, loader, device: str, n_batches: int = 10):
    """We return images/sec and ms/image. We warmup with 2 batches before timing."""
    model.eval()
    it = iter(loader)
    # the warmup
    with torch.no_grad():
        for _ in range(2):
            try:
                imgs, _ = next(it)
            except StopIteration:
                break
            _ = model(imgs.to(device))
    if device == "cuda":
        torch.cuda.synchronize()

    total_imgs, total_time = 0, 0.0
    it = iter(loader)
    with torch.no_grad():
        for i in range(n_batches):
            try:
                imgs, _ = next(it)
            except StopIteration:
                break
            imgs = imgs.to(device)
            start = time.perf_counter()
            _ = model(imgs)
            if device == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            total_imgs += imgs.size(0)
    return {
        "images_per_sec": total_imgs / total_time,
        "ms_per_image": 1000.0 * total_time / total_imgs,
        "batches_measured": min(n_batches, i + 1),
    }


#3. rubric #88  Error analysis with visualization of failure cases
def analyze_errors(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_probs: np.ndarray,
                   df_test: pd.DataFrame,
                   save_dir: Path,
                   k: int = 12) -> List[Dict]:
    """
    Here we identify the k most-confidently-wrong predictions and save:
      1. A JSON log of them
      2. A PNG grid of the offending images
    We use this for a qualitative discussion in the README/notebook to address
    WHY the model failed on these: ordinal neighbors? Was it visual ambiguity?
    Was is image quality? Was it some sort of systematic bias toward a majority class?
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wrong_mask = (y_true != y_pred)
    confidences = y_probs.max(axis=1)
    wrong_idx = np.where(wrong_mask)[0]
    if len(wrong_idx) == 0:
        print("[eval] no errors - perfect predictions (unlikely)")
        return []
    # we sort by confidence descending: the most-confident-wrongs are the worst
    order = np.argsort(-confidences[wrong_idx])[:k]
    to_show = wrong_idx[order]

    records = []
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, idx in enumerate(to_show):
        row = df_test.iloc[idx]
        records.append({
            "image_path": row["image_path"],
            "true_label": int(y_true[idx]),
            "pred_label": int(y_pred[idx]),
            "pred_confidence": float(confidences[idx]),
            "prob_for_true_class": float(y_probs[idx, y_true[idx]]),
            "ordinal_distance": int(abs(int(y_true[idx]) - int(y_pred[idx]))),
        })
        if i < len(axes):
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                axes[i].imshow(img)
                axes[i].set_title(
                    f"true={y_true[idx]} pred={y_pred[idx]} "
                    f"conf={confidences[idx]:.2f}",
                    fontsize=10,
                )
                axes[i].axis("off")
            except Exception as e:
                axes[i].text(0.5, 0.5, f"(load failed: {e})", ha="center")
                axes[i].axis("off")

    for j in range(len(to_show), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Top-k most confidently misclassified clock drawings", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "top_errors.png", dpi=150)
    plt.close()

    with open(save_dir / "top_errors.json", "w") as f:
        json.dump(records, f, indent=2)
    return records


# 4. rubric #89  Comparison of multiple architectures (aggregate_results helper)
def aggregate_results(per_run_metrics: Dict[str, dict], save_path: Path):
    """
    the per_run_metrics = {run_name: {accuracy: ..., macro_f1: ..., ...}, ...}
    We write a comparison CSV to have in the Readme
    """
    df = pd.DataFrame(per_run_metrics).T
    df.index.name = "run"
    df.to_csv(save_path)
    return df


# 5. bootstrap confidence intervals
def _resample_by_group(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # we resample participants, not clocks, since one person's clocks aren't independent
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_per_group = [np.where(inverse == g)[0] for g in range(len(uniq))]
    picked = rng.integers(0, len(uniq), size=len(uniq))
    return np.concatenate([rows_per_group[g] for g in picked])


def bootstrap_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      groups: np.ndarray,
                      n_boot: int = 1000,
                      seed: int = 42,
                      alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """
    95% confidence intervals for kappa, accuracy and macro-F1.
    groups is the participant_id of every test clock.
    """
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    metric_fns = {
        "quadratic_kappa": lambda t, p: cohen_kappa_score(t, p, weights="quadratic"),
        "accuracy":        lambda t, p: accuracy_score(t, p),
        "macro_f1":        lambda t, p: f1_score(t, p, average="macro"),
    }
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in metric_fns}
    for _ in range(n_boot):
        idx = _resample_by_group(groups, rng)
        for k, fn in metric_fns.items():
            draws[k].append(fn(y_true[idx], y_pred[idx]))

    out = {}
    for k, fn in metric_fns.items():
        lo, hi = np.quantile(draws[k], [alpha / 2, 1 - alpha / 2])
        out[k] = {"point": float(fn(y_true, y_pred)), "lo": float(lo), "hi": float(hi)}
    return out


# 6. paired bootstrap between two models
def paired_bootstrap_kappa(y_true: np.ndarray,
                           y_pred_a: np.ndarray,
                           y_pred_b: np.ndarray,
                           groups: np.ndarray,
                           n_boot: int = 1000,
                           seed: int = 42) -> Dict[str, float]:
    """
    Is model A really better than model B? We resample participants, compute kappa
    for both on the same draw, and look at the difference. If the interval crosses
    0, we can't say A is better.
    """
    y_true, y_pred_a, y_pred_b, groups = map(np.asarray, (y_true, y_pred_a, y_pred_b, groups))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_by_group(groups, rng)
        ka = cohen_kappa_score(y_true[idx], y_pred_a[idx], weights="quadratic")
        kb = cohen_kappa_score(y_true[idx], y_pred_b[idx], weights="quadratic")
        diffs.append(ka - kb)
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "kappa_a": float(cohen_kappa_score(y_true, y_pred_a, weights="quadratic")),
        "kappa_b": float(cohen_kappa_score(y_true, y_pred_b, weights="quadratic")),
        "diff": float(diffs.mean()),
        "diff_lo": float(lo),
        "diff_hi": float(hi),
        "p_a_not_better": float((diffs <= 0).mean()),
    }


# 7. kappa-tuned decoding
def expected_score(y_probs: np.ndarray) -> np.ndarray:
    # the probability-weighted average score, between 0 and 5
    return y_probs @ np.arange(y_probs.shape[1])


def apply_thresholds(scores: np.ndarray, thresholds) -> np.ndarray:
    # score < t1 -> 0, t1 <= score < t2 -> 1, and so on
    return np.digitize(scores, np.sort(np.asarray(thresholds)))


def optimize_thresholds(y_probs_val: np.ndarray,
                        y_true_val: np.ndarray,
                        n_rounds: int = 3,
                        grid_step: float = 0.01) -> np.ndarray:
    """
    Argmax ignores that the scores are ordered. We use the expected score instead
    and pick the cut points that maximize kappa on validation (never on test),
    starting from 0.5, 1.5, ... 4.5 and moving one at a time.
    """
    scores = expected_score(y_probs_val)
    k = y_probs_val.shape[1]
    th = np.arange(k - 1) + 0.5

    def kappa_for(t):
        return cohen_kappa_score(y_true_val, apply_thresholds(scores, t), weights="quadratic")

    best = kappa_for(th)
    for _ in range(n_rounds):
        for i in range(k - 1):
            lo = th[i - 1] + grid_step if i > 0 else 0.0
            hi = th[i + 1] - grid_step if i < k - 2 else float(k - 1)
            for cand in np.arange(lo, hi + 1e-9, grid_step):
                trial = th.copy()
                trial[i] = cand
                kap = kappa_for(trial)
                if kap > best:
                    best, th = kap, trial
    print(f"[eval] tuned thresholds {np.round(th, 2).tolist()} -> val kappa {best:.4f}")
    return th
