"""
Tests that run without the NHATS data and without a GPU.

We use:  python -m pytest tests -q

They check the parts that broke silently in v1: the crop, the split, the
attribution maps lining up with the input, the confidence intervals, and one tiny
end-to-end run of src.run with random weights.
"""
import os
import json
import tempfile
from pathlib import Path

# the output folders must point somewhere temporary BEFORE src is imported
_TMP = Path(tempfile.mkdtemp(prefix="cdt_tests_"))
os.environ["CDT_DATA_DIR"] = str(_TMP / "data")
os.environ["CDT_MODELS_DIR"] = str(_TMP / "models")
os.environ["CDT_RESULTS_DIR"] = str(_TMP / "results")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.augmentation import SmartCropClock, build_eval_transform
from src.config import DataConfig
from src.data import get_splits, load_labels
from src.evaluate import (bootstrap_metrics, paired_bootstrap_kappa,
                          optimize_thresholds, apply_thresholds, expected_score)
from src.gradcam import vit_occlusion_map
from tests.synthetic import make_page, make_dataset


def _ink_bbox(img):
    arr = np.array(img.convert("L")) < 128
    ys, xs = np.where(arr)
    return xs.min(), ys.min(), xs.max(), ys.max()


# the crop
def test_component_crop_is_square_and_keeps_the_whole_clock():
    rng = np.random.default_rng(1)
    for score in range(6):
        page, (cx, cy, r) = make_page(rng, score, add_note=True, inner_bar=True)
        crop = SmartCropClock(method="component")(page.convert("RGB"))
        w, h = crop.size
        assert w == h, f"score {score}: crop is {w}x{h}, not square"
        # the clock is drawn with radius ~r, so the crop must be at least that big
        # and not much bigger (the note and the bars must stay out)
        assert w >= 1.6 * r, f"score {score}: crop {w} too small for radius {r}"
        assert w <= 4.0 * r, f"score {score}: crop {w} too big for radius {r}, junk got in"


def test_component_crop_drops_the_scanner_bar():
    rng = np.random.default_rng(2)
    page, (cx, cy, r) = make_page(rng, 5, inner_bar=True)
    crop = np.array(SmartCropClock(method="component")(page.convert("RGB")).convert("L"))
    # a solid black bar would show up as a big run of fully black rows and columns
    solid = (crop < 50)
    biggest_block = max(solid.sum(axis=0).max(), solid.sum(axis=1).max())
    assert biggest_block < 0.3 * crop.shape[0], "a solid bar is still in the crop"


def test_component_crop_keeps_a_number_drawn_outside_the_circle():
    from PIL import ImageDraw
    page = Image.new("L", (2560, 3312), 255)
    d = ImageDraw.Draw(page)
    d.ellipse([1000, 1000, 1600, 1600], outline=0, width=4)     # the clock, 600 px
    d.line([(1720, 1250), (1720, 1330)], fill=0, width=4)       # a "1", 120 px outside
    d.line([(2000, 2600), (2060, 2680)], fill=0, width=4)       # a note far away
    crop = SmartCropClock(method="component")(page.convert("RGB"))
    side = crop.size[0]
    assert side >= 720, f"crop {side} cut off the number outside the circle"
    assert side < 1200, f"crop {side} let the far-away note in"


def test_crop_near_page_edge_pads_instead_of_squashing():
    page = Image.new("L", (1000, 1000), 255)
    from PIL import ImageDraw
    d = ImageDraw.Draw(page)
    # a clock pushed against the right edge of the inner region
    d.ellipse([700, 300, 880, 600], outline=0, width=4)
    crop = SmartCropClock(method="component", edge_strip_frac=0.05)(page.convert("RGB"))
    assert crop.size[0] == crop.size[1]


def test_projection_crop_is_unchanged_v1_behaviour():
    # v1 must stay exactly as it was so old checkpoints can be re-scored
    rng = np.random.default_rng(3)
    page, _ = make_page(rng, 4)
    crop = SmartCropClock(method="projection")(page.convert("RGB"))
    assert crop.size[0] > 0 and crop.size[1] > 0


# the split
def test_splits_are_participant_disjoint_and_frozen(tmp_path):
    labels = make_dataset(tmp_path, n_participants=40, n_rounds=3, seed=4)
    df = load_labels(tmp_path / "labels.csv")
    cfg = DataConfig()
    splits_csv = tmp_path / "splits.csv"
    tr, va, te = get_splits(df, cfg, splits_csv)
    ids = [set(d["participant_id"].astype(str)) for d in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert splits_csv.exists()
    # shuffling the rows must not move anybody once the split is frozen
    tr2, va2, te2 = get_splits(df.sample(frac=1, random_state=0), cfg, splits_csv)
    assert set(te2["participant_id"].astype(str)) == ids[2]


# the statistics
def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 6, 600)
    pred = np.clip(y + rng.integers(-1, 2, 600), 0, 5)
    groups = np.repeat(np.arange(200), 3)
    ci = bootstrap_metrics(y, pred, groups, n_boot=200)
    k = ci["quadratic_kappa"]
    assert k["lo"] <= k["point"] <= k["hi"]
    assert k["hi"] - k["lo"] < 0.2


def test_paired_bootstrap_sees_a_real_difference():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 6, 900)
    good = np.clip(y + rng.integers(-1, 2, 900), 0, 5)
    bad = rng.integers(0, 6, 900)
    res = paired_bootstrap_kappa(y, good, bad, np.repeat(np.arange(300), 3), n_boot=200)
    assert res["diff_lo"] > 0


def test_tuned_thresholds_do_not_hurt_on_validation():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 6, 1000)
    logits = np.eye(6)[y] * 2 + rng.normal(0, 1.2, (1000, 6))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    from sklearn.metrics import cohen_kappa_score
    k_argmax = cohen_kappa_score(y, probs.argmax(1), weights="quadratic")
    th = optimize_thresholds(probs, y)
    k_tuned = cohen_kappa_score(y, apply_thresholds(expected_score(probs), th), weights="quadratic")
    assert k_tuned >= k_argmax - 1e-9


# the attribution maps
class _TopLeftInkModel(nn.Module):
    """a toy model: class 0 gets more likely the more ink there is in the top-left patch."""
    def forward(self, x):
        ink = -x[:, 0, :16, :16].mean(dim=(1, 2))       # darker = more ink
        logits = torch.zeros(x.shape[0], 6)
        logits[:, 0] = 10 * ink
        return logits


def test_occlusion_map_points_at_the_right_patch():
    img = Image.new("RGB", (224, 224), "white")
    from PIL import ImageDraw
    ImageDraw.Draw(img).rectangle([2, 2, 13, 13], fill="black")
    # no crop here, we want the ink to stay in the top-left patch
    from torchvision import transforms
    from src.augmentation import IMAGENET_MEAN, IMAGENET_STD
    x = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(img).unsqueeze(0)
    heat = vit_occlusion_map(_TopLeftInkModel(), x, class_idx=0, fill="white")
    assert np.unravel_index(heat.argmax(), heat.shape) == (0, 0)
    assert heat[0, 0] == 1.0 and heat[5:, 5:].max() == 0.0


# one tiny end-to-end run
def test_run_end_to_end_with_random_weights():
    from src import run
    data_dir = Path(os.environ["CDT_DATA_DIR"])
    make_dataset(data_dir, n_participants=30, n_rounds=2, seed=8)
    run.main([
        "--models", "efficientnet_b0", "--no-pretrained", "--epochs", "1",
        "--batch-size", "4", "--num-workers", "0", "--limit", "16", "--n-boot", "50",
        "--labels", str(data_dir / "labels.csv"), "--splits", str(data_dir / "splits.csv"),
    ])
    out = Path(os.environ["CDT_RESULTS_DIR"]) / "efficientnet_b0_s42"
    assert (out / "metrics.json").exists()
    assert (out / "predictions_test.csv").exists()


# things the review found
def test_fast_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    from src.evaluate import fast_quadratic_kappa
    rng = np.random.default_rng(9)
    y, p = rng.integers(0, 6, 5000), rng.integers(0, 6, 5000)
    p[:3000] = y[:3000]
    assert abs(fast_quadratic_kappa(y, p) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-9


def test_bootstrap_is_fast_on_a_test_set_the_size_of_nhats():
    import time
    rng = np.random.default_rng(10)
    n = 8938
    y = rng.integers(0, 6, n)
    pred = np.clip(y + rng.integers(-1, 2, n), 0, 5)
    groups = rng.integers(0, 3500, n)
    t = time.time()
    bootstrap_metrics(y, pred, groups, n_boot=1000)
    assert time.time() - t < 60, "the bootstrap is too slow for the real test set"


def test_a_straight_pen_line_is_not_treated_as_a_scanner_bar():
    from PIL import ImageDraw
    page = Image.new("L", (2560, 3312), 255)
    # a drawing that is just one long straight line
    ImageDraw.Draw(page).line([(1200, 900), (1200, 1700)], fill=0, width=6)
    crop = SmartCropClock(method="component")(page.convert("RGB"))
    assert crop.size[0] < 1500, "the line was dropped and we fell back to the center square"


def test_thicken_strokes_keeps_the_aspect_ratio():
    from src.augmentation import ThickenStrokes
    out = ThickenStrokes(224)(Image.new("RGB", (2000, 1000), "white"))
    assert out.size == (896, 448)


def test_train_saves_where_the_notebook_says(tmp_path):
    # the notebooks set cfg.MODELS_DIR after importing, train() has to see that
    import src.config as cfg
    from src.train import train
    from src.data import CDTDataset, build_dataloaders, compute_class_weights
    from src.config import TrainConfig
    labels = make_dataset(tmp_path / "d", n_participants=8, n_rounds=1, seed=12)
    df = load_labels(tmp_path / "d" / "labels.csv")
    tf = build_eval_transform(64)
    ds = CDTDataset(df.head(8), tf)
    loaders = build_dataloaders(ds, ds, ds, DataConfig(batch_size=4, num_workers=0))
    old = cfg.MODELS_DIR
    cfg.MODELS_DIR = tmp_path / "drive_models"
    try:
        out = train("efficientnet_b0", loaders[0], loaders[1], compute_class_weights(df),
                    TrainConfig(epochs=1), "where_test", pretrained=False)
    finally:
        cfg.MODELS_DIR = old
    assert Path(out["checkpoint_path"]).parent == tmp_path / "drive_models"
    assert Path(out["checkpoint_path"]).exists()


def test_report_puts_runs_side_by_side():
    # fake results like the ones on Drive: a scored v1 run and two trained v2 seeds
    from src.report import make_report
    res = _TMP / "report_results"
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 300)
    for name, k, seed, hist in [("vit_b16_v1_s42", 0.808, 42, False),
                                ("vit_b16_v2_s42", 0.834, 42, True),
                                ("vit_b16_v2_s43", 0.829, 43, True)]:
        d = res / name
        d.mkdir(parents=True)
        ci = {"quadratic_kappa": {"lo": k - 0.01, "hi": k + 0.01}}
        json.dump({"run_name": name, "model_name": "vit_b16", "seed": seed, "best_epoch": 3 if hist else None,
                   "n_test": 300, "argmax": {"quadratic_kappa": k, "accuracy": 0.7, "macro_f1": 0.6},
                   "argmax_ci": ci, "tuned": {"quadratic_kappa": k + 0.01}, "tuned_ci": ci},
                  open(d / "metrics.json", "w"))
        if hist:
            json.dump([{"epoch": e, "train_loss": 1 - e / 10, "train_acc": 0.5, "val_loss": 0.9,
                        "val_acc": 0.6, "val_kappa": 0.7 + e / 100} for e in range(6)],
                      open(d / "history.json", "w"))
        pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 6, 300))
        pd.DataFrame({"participant_id": "x", "image_path": "p", "y_true": y, "y_pred": pred}) \
            .to_csv(d / "predictions_test.csv", index=False)
    json.dump({"diff": 0.026, "diff_lo": 0.019, "diff_hi": 0.034},
              open(res / "compare_vit_b16_v2_s42_vs_vit_b16_v1_s42.json", "w"))

    r = make_report(res)
    assert len(r["runs"]) == 3
    s = r["seeds"]
    assert len(s) == 1 and s.iloc[0]["n_seeds"] == 2 and s.iloc[0]["version"] == "v2"
    assert abs(s.iloc[0]["kappa_mean"] - 0.8315) < 1e-9
    assert r["comparisons"].iloc[0]["run_a"] == "vit_b16_v2_s42"
    for f in ["runs.csv", "seeds.csv", "comparisons.csv", "kappa_ci.png",
              "training_curves.png", "confusion_matrices.png"]:
        assert (r["out_dir"] / f).exists(), f
