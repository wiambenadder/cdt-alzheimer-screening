"""
Puts all the runs side by side. Every run already saves its own files on Drive
(metrics.json, history.json, predictions_test.csv), so this only reads them back:
it needs no GPU and no images, and works after any disconnect.

    python -m src.report                      # reads RESULTS_DIR, writes RESULTS_DIR/report
    python -m src.report --results /path/to/results_v2

Everything it writes is aggregate (scores, curves, confusion matrices), no participant
IDs and no drawings, so it can go in the report or on slides.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .config import NUM_CLASSES, RESULTS_DIR


def load_runs(results_dir: Path) -> pd.DataFrame:
    """One row per run, from its metrics.json."""
    rows = []
    for f in sorted(Path(results_dir).glob("*/metrics.json")):
        m = json.load(open(f))
        name = m["run_name"]
        ci, tci = m["argmax_ci"]["quadratic_kappa"], m["tuned_ci"]["quadratic_kappa"]
        rows.append({
            "run": name,
            "model": m.get("model_name", name.split("_v")[0]),
            "version": "v2" if "_v2" in name else "v1" if "_v1" in name else "other",
            "seed": m.get("seed"),
            "best_epoch": m.get("best_epoch"),
            "n_test": m["n_test"],
            "kappa": m["argmax"]["quadratic_kappa"], "kappa_lo": ci["lo"], "kappa_hi": ci["hi"],
            "tuned_kappa": m["tuned"]["quadratic_kappa"], "tuned_lo": tci["lo"], "tuned_hi": tci["hi"],
            "accuracy": m["argmax"].get("accuracy"),
            "macro_f1": m["argmax"].get("macro_f1"),
        })
    return pd.DataFrame(rows)


def load_comparisons(results_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(results_dir).glob("compare_*.json")):
        c = json.load(open(f))
        a, b = f.stem[len("compare_"):].split("_vs_")
        rows.append({"run_a": a, "run_b": b, "diff": c["diff"],
                     "diff_lo": c["diff_lo"], "diff_hi": c["diff_hi"]})
    return pd.DataFrame(rows)


def seed_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread over the seeds of each model and version (only where there are 2+ seeds)."""
    out = []
    for (model, version), g in runs.groupby(["model", "version"]):
        if len(g) < 2:
            continue
        out.append({
            "model": model, "version": version, "n_seeds": len(g),
            "seeds": " ".join(str(s) for s in g["seed"]),
            "kappa_mean": g["kappa"].mean(), "kappa_sd": g["kappa"].std(ddof=1),
            "tuned_mean": g["tuned_kappa"].mean(), "tuned_sd": g["tuned_kappa"].std(ddof=1),
            "acc_mean": g["accuracy"].mean(), "f1_mean": g["macro_f1"].mean(),
        })
    return pd.DataFrame(out)


def plot_kappas(runs: pd.DataFrame, path: Path):
    """Every run with its 95% CI, plain and with the tuned cut points."""
    runs = runs.sort_values(["version", "model", "seed"]).reset_index(drop=True)
    y = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(runs) + 2))
    ax.errorbar(runs["kappa"], y - 0.12,
                xerr=[runs["kappa"] - runs["kappa_lo"], runs["kappa_hi"] - runs["kappa"]],
                fmt="o", color="#1b3557", capsize=3, label="kappa")
    ax.errorbar(runs["tuned_kappa"], y + 0.12,
                xerr=[runs["tuned_kappa"] - runs["tuned_lo"], runs["tuned_hi"] - runs["tuned_kappa"]],
                fmt="s", color="#c0392b", capsize=3, label="kappa, tuned cut points")
    ax.set_yticks(y, runs["run"])
    ax.invert_yaxis()
    ax.set_xlabel("quadratic weighted kappa on the test clocks (95% CI)")
    ax.grid(axis="x", alpha=0.3)
    # under the plot, so it never covers a run
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(results_dir: Path, runs: pd.DataFrame, path: Path) -> bool:
    """Validation kappa and loss per epoch, one line per run that was trained (not only scored)."""
    hist = {}
    for name in runs["run"]:
        f = Path(results_dir) / name / "history.json"
        if f.exists():
            hist[name] = pd.DataFrame(json.load(open(f)))
    if not hist:
        return False
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, h in hist.items():
        best = runs.loc[runs["run"] == name, "best_epoch"].iloc[0]
        line, = a.plot(h["epoch"], h["val_kappa"], marker="o", ms=3, label=name)
        b.plot(h["epoch"], h["val_loss"], marker="o", ms=3, color=line.get_color(), label=name)
        if pd.notna(best):
            # the star is the epoch that was kept
            row = h[h["epoch"] == int(best)]
            a.plot(row["epoch"], row["val_kappa"], "*", ms=14, color=line.get_color())
            b.plot(row["epoch"], row["val_loss"], "*", ms=14, color=line.get_color())
    a.set_title("validation kappa (higher is better)")
    b.set_title("validation loss (lower is better)")
    for ax in (a, b):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.suptitle("star = the epoch that was kept", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_confusions(results_dir: Path, runs: pd.DataFrame, path: Path) -> bool:
    """Row-normalized confusion matrices (recall per true score), side by side."""
    mats = []
    for name in runs["run"]:
        f = Path(results_dir) / name / "predictions_test.csv"
        if f.exists():
            p = pd.read_csv(f, usecols=["y_true", "y_pred"])
            cm = confusion_matrix(p["y_true"], p["y_pred"], labels=range(NUM_CLASSES)).astype(float)
            cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
            mats.append((name, cm))
    if not mats:
        return False
    fig, axes = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 3.8), squeeze=False)
    for ax, (name, cm) in zip(axes[0], mats):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("predicted score")
        ax.set_ylabel("true score")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_results_md(runs, comps, seeds, results_dir: Path, path: Path):
    """The same tables as text, each number with the file it was read from."""
    from .record import now
    L = ["# Results", "", f"Written automatically on {now()} by `python -m src.report` from the "
         f"metrics.json and compare_*.json files in `{results_dir}`. Nothing here is typed by hand.", "",
         "## Every run", "", "| run | test clocks | best epoch | kappa (95% CI) | tuned kappa (95% CI) | accuracy | macro F1 | source |",
         "|---|---|---|---|---|---|---|---|"]
    for r in runs.itertuples():
        best = "" if pd.isna(r.best_epoch) else int(r.best_epoch)
        L.append(f"| {r.run} | {r.n_test:,} | {best} | {r.kappa:.3f} [{r.kappa_lo:.3f}, {r.kappa_hi:.3f}] | "
                 f"{r.tuned_kappa:.3f} [{r.tuned_lo:.3f}, {r.tuned_hi:.3f}] | {r.accuracy:.3f} | {r.macro_f1:.3f} | "
                 f"`{r.run}/metrics.json` |")
    if not seeds.empty:
        L += ["", "## Mean over the seeds", "", "| model | version | seeds | kappa mean (sd) | tuned mean (sd) |", "|---|---|---|---|---|"]
        for s in seeds.itertuples():
            L.append(f"| {s.model} | {s.version} | {s.seeds} | {s.kappa_mean:.3f} ({s.kappa_sd:.3f}) | "
                     f"{s.tuned_mean:.3f} ({s.tuned_sd:.3f}) |")
    if not comps.empty:
        L += ["", "## Paired comparisons (same test clocks)", "", "| A | B | kappa A - kappa B (95% CI) | source |", "|---|---|---|---|"]
        for c in comps.itertuples():
            L.append(f"| {c.run_a} | {c.run_b} | {c.diff:+.3f} [{c.diff_lo:+.3f}, {c.diff_hi:+.3f}] | "
                     f"`compare_{c.run_a}_vs_{c.run_b}.json` |")
        L += ["", "If the interval doesn't cross 0, the difference is real (not noise)."]
    Path(path).write_text("\n".join(L) + "\n")


def make_report(results_dir: Path = RESULTS_DIR, out_dir: Path = None) -> dict:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    if runs.empty:
        print(f"[report] no runs in {results_dir}")
        return {}
    comps = load_comparisons(results_dir)
    seeds = seed_summary(runs)

    runs.to_csv(out_dir / "runs.csv", index=False)
    comps.to_csv(out_dir / "comparisons.csv", index=False)
    seeds.to_csv(out_dir / "seeds.csv", index=False)
    plot_kappas(runs, out_dir / "kappa_ci.png")
    has_curves = plot_training_curves(results_dir, runs, out_dir / "training_curves.png")
    has_cm = plot_confusions(results_dir, runs, out_dir / "confusion_matrices.png")

    write_results_md(runs, comps, seeds, results_dir, out_dir / "RESULTS.md")

    if runs["n_test"].nunique() > 1:
        print("[report] careful: the runs were not all tested on the same number of clocks")
    print(f"[report] {len(runs)} runs -> {out_dir}")
    return {"runs": runs, "comparisons": comps, "seeds": seeds, "out_dir": out_dir,
            "figures": [p for p, ok in ((out_dir / "kappa_ci.png", True),
                                         (out_dir / "training_curves.png", has_curves),
                                         (out_dir / "confusion_matrices.png", has_cm)) if ok]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    r = make_report(args.results, args.out)
    if r:
        pd.set_option("display.width", 200)
        print(r["runs"][["run", "n_test", "best_epoch", "kappa", "kappa_lo", "kappa_hi",
                         "tuned_kappa", "accuracy", "macro_f1"]].round(3).to_string(index=False))
        if not r["seeds"].empty:
            print(r["seeds"].round(3).to_string(index=False))
        if not r["comparisons"].empty:
            print(r["comparisons"].round(3).to_string(index=False))
