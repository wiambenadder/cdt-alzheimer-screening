"""
One entry point for every experiment.

Examples (from the repo root):
    # the main ViT run, repeated with 3 seeds
    python -m src.run --models vit_b16 --seeds 42 43 44

    # the 3-architecture comparison
    python -m src.run --models vgg16 efficientnet_b0 vit_b16

    # re-score a checkpoint that already exists (no training), with CIs and tuned cut points
    python -m src.run --models vit_b16 --evaluate-only --checkpoint models/vit_b16_ft_best.pt

    # is run A really better than run B? (both need predictions_test.csv)
    python -m src.run --compare vit_b16_s42 vgg16_s42

On Colab, point the data and model folders at Drive first:
    %env CDT_DATA_DIR=/content/drive/MyDrive/cdt-data
    %env CDT_MODELS_DIR=/content/drive/MyDrive/cdt-data/models

Every run writes metrics.json, predictions_val.csv and predictions_test.csv
to results/<run_name>/.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    DataConfig, TrainConfig, AugConfig, RESULTS_DIR, LABELS_CSV, SPLITS_CSV,
    NUM_CLASSES, SEED,
)
from .data import (
    load_labels, get_splits, compute_class_weights, make_weighted_sampler,
    CDTDataset, build_dataloaders,
)
from .augmentation import build_train_transform, build_eval_transform
from .models import get_model, MODEL_INPUT_SIZE
from .train import train
from .evaluate import (
    collect_predictions, compute_all_metrics, bootstrap_metrics,
    optimize_thresholds, apply_thresholds, expected_score, paired_bootstrap_kappa,
    save_classification_report, plot_confusion_matrix,
)
from .utils import get_device, load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train and evaluate CDT scoring models")
    p.add_argument("--models", nargs="+", default=["vit_b16"])
    p.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    p.add_argument("--run-tag", default="", help="extra text added to every run name")

    # data
    p.add_argument("--labels", type=Path, default=LABELS_CSV)
    p.add_argument("--splits", type=Path, default=SPLITS_CSV)
    p.add_argument("--crop-method", default="component", choices=["component", "projection"])
    p.add_argument("--thicken-strokes", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--imbalance", default="class_weights",
                   choices=["class_weights", "weighted_sampler", "none"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="use only this many clocks per split (smoke tests)")

    # training
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--select-metric", default="val_loss", choices=["val_loss", "val_kappa"])
    p.add_argument("--no-pretrained", action="store_true",
                   help="random weights, only for offline smoke tests")

    # evaluation
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None)
    return p.parse_args(argv)


def predictions_frame(df, y_true, y_pred, y_probs):
    out = pd.DataFrame({
        "participant_id": df["participant_id"].astype(str).values,
        "round": df["round"].values if "round" in df.columns else -1,
        "image_path": df["image_path"].values,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    for c in range(NUM_CLASSES):
        out[f"p{c}"] = y_probs[:, c]
    return out


def evaluate_run(model, val_loader, test_loader, val_df, test_df, run_dir, device, n_boot):
    yv, pv, probs_v = collect_predictions(model, val_loader, device)
    yt, pt, probs_t = collect_predictions(model, test_loader, device)
    predictions_frame(val_df, yv, pv, probs_v).to_csv(run_dir / "predictions_val.csv", index=False)
    predictions_frame(test_df, yt, pt, probs_t).to_csv(run_dir / "predictions_test.csv", index=False)

    groups = test_df["participant_id"].astype(str).values
    metrics = {"argmax": compute_all_metrics(yt, pt, probs_t)}
    metrics["argmax_ci"] = bootstrap_metrics(yt, pt, groups, n_boot=n_boot)

    # cut points are picked on val and only applied to test
    th = optimize_thresholds(probs_v, yv)
    pt_tuned = apply_thresholds(expected_score(probs_t), th)
    metrics["tuned_thresholds"] = th.tolist()
    metrics["tuned"] = compute_all_metrics(yt, pt_tuned, probs_t)
    metrics["tuned_ci"] = bootstrap_metrics(yt, pt_tuned, groups, n_boot=n_boot)

    save_classification_report(yt, pt, run_dir / "classification_report.json")
    plot_confusion_matrix(yt, pt, run_dir / "confusion_matrix_normalized.png", normalize=True)

    k, ci = metrics["argmax"]["quadratic_kappa"], metrics["argmax_ci"]["quadratic_kappa"]
    kt, cit = metrics["tuned"]["quadratic_kappa"], metrics["tuned_ci"]["quadratic_kappa"]
    print(f"[eval] test kappa (argmax) = {k:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]")
    print(f"[eval] test kappa (tuned)  = {kt:.3f}  [{cit['lo']:.3f}, {cit['hi']:.3f}]")
    return metrics


def compare(run_a: str, run_b: str, n_boot: int):
    a = pd.read_csv(RESULTS_DIR / run_a / "predictions_test.csv", dtype={"participant_id": str})
    b = pd.read_csv(RESULTS_DIR / run_b / "predictions_test.csv", dtype={"participant_id": str})
    assert (a["image_path"].values == b["image_path"].values).all(), \
        "the two runs were scored on different test sets"
    res = paired_bootstrap_kappa(a["y_true"].values, a["y_pred"].values,
                                 b["y_pred"].values, a["participant_id"].values, n_boot=n_boot)
    print(f"[compare] {run_a}: kappa {res['kappa_a']:.3f}   {run_b}: kappa {res['kappa_b']:.3f}")
    print(f"[compare] difference {res['diff']:+.3f}  95% CI [{res['diff_lo']:+.3f}, {res['diff_hi']:+.3f}]")
    if res["diff_lo"] <= 0 <= res["diff_hi"]:
        print("[compare] the interval crosses 0, so we can't say one is better")
    out = RESULTS_DIR / f"compare_{run_a}_vs_{run_b}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    return res


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        compare(*args.compare, n_boot=args.n_boot)
        return

    data_cfg = DataConfig(batch_size=args.batch_size, num_workers=args.num_workers,
                          imbalance_strategy=args.imbalance, crop_method=args.crop_method,
                          thicken_strokes=args.thicken_strokes)
    train_cfg = TrainConfig(epochs=args.epochs, early_stopping_patience=args.patience,
                            select_metric=args.select_metric)
    aug_cfg = AugConfig()

    df = load_labels(args.labels)
    train_df, val_df, test_df = get_splits(df, data_cfg, args.splits)
    if args.limit:
        train_df, val_df, test_df = (d.head(args.limit) for d in (train_df, val_df, test_df))

    # one correction at a time, using both would double count the rare classes
    if data_cfg.imbalance_strategy == "class_weights":
        class_weights, sampler = compute_class_weights(train_df), None
    elif data_cfg.imbalance_strategy == "weighted_sampler":
        class_weights, sampler = torch.ones(NUM_CLASSES), make_weighted_sampler(train_df)
    else:
        class_weights, sampler = torch.ones(NUM_CLASSES), None
    print(f"[run] imbalance={data_cfg.imbalance_strategy} weights={np.round(class_weights.tolist(), 3).tolist()}")

    device = get_device()
    for model_name in args.models:
        size = MODEL_INPUT_SIZE[model_name]
        train_tf = build_train_transform(size, aug_cfg, use_aug=not args.no_aug,
                                         crop_method=data_cfg.crop_method,
                                         thicken_strokes=data_cfg.thicken_strokes)
        eval_tf = build_eval_transform(size, data_cfg.crop_method, data_cfg.thicken_strokes)
        loaders = build_dataloaders(CDTDataset(train_df, train_tf), CDTDataset(val_df, eval_tf),
                                    CDTDataset(test_df, eval_tf), data_cfg, sampler=sampler)
        train_loader, val_loader, test_loader = loaders

        for seed in args.seeds:
            run_name = f"{model_name}{'_frozen' if args.freeze_backbone else ''}" \
                       f"{'_noaug' if args.no_aug else ''}{args.run_tag}_s{seed}"
            run_dir = RESULTS_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            if args.evaluate_only:
                if args.checkpoint is None:
                    raise SystemExit("--evaluate-only needs --checkpoint")
                ckpt = args.checkpoint
                summary = {"run_name": run_name, "checkpoint_path": str(ckpt)}
            else:
                summary = train(model_name, train_loader, val_loader, class_weights,
                                train_cfg, run_name, freeze_backbone=args.freeze_backbone,
                                seed=seed, pretrained=not args.no_pretrained)
                ckpt = summary["checkpoint_path"]

            model, _, _ = get_model(model_name, freeze_backbone=args.freeze_backbone,
                                    dropout=train_cfg.dropout, pretrained=False)
            model = load_checkpoint(model, ckpt, device).to(device)
            metrics = evaluate_run(model, val_loader, test_loader, val_df, test_df,
                                   run_dir, device, args.n_boot)

            record = {
                "run_name": run_name, "model_name": model_name, "seed": seed,
                "data_config": asdict(data_cfg), "train_config": asdict(train_cfg),
                "freeze_backbone": args.freeze_backbone, "use_aug": not args.no_aug,
                "best_epoch": summary.get("best_epoch"), "checkpoint": str(ckpt),
                "n_test": len(test_df), **metrics,
            }
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(record, f, indent=2)
            print(f"[run] wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
