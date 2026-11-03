"""
The paper trail of every run, so every number we report can be traced back to a file.

For each run, next to metrics.json, we keep:
  train_log.txt   everything the run printed, from every session (a disconnect doesn't erase it)
  sessions.json   one entry per Colab session that worked on the run: when it started and ended,
                  the code version (git commit), the GPU and the exact command
  RUN_RECORD.md   a readable summary written only from the files above, never typed by hand

    python -m src.record vit_b16_v2_s43      # (re)write RUN_RECORD.md of a run from its files
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import RESULTS_DIR

REPO = Path(__file__).resolve().parent.parent


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment() -> dict:
    return {
        "git_commit": git_commit(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class RunLog:
    """Copies everything printed into run_dir/train_log.txt (added at the end, never
    overwritten) and adds this session to sessions.json."""

    def __init__(self, run_dir: Path, command: str, note: str = ""):
        self.run_dir = Path(run_dir)
        self.command = command
        self.note = note

    def __enter__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.session = {"started": now(), "ended": None, "status": "running",
                        "command": self.command, "note": self.note, **environment()}
        self.sessions = load_sessions(self.run_dir)
        self.sessions.append(self.session)
        save_sessions(self.run_dir, self.sessions)

        self.file = open(self.run_dir / "train_log.txt", "a", buffering=1)
        self.file.write(f"\n===== session started {self.session['started']} | "
                        f"{self.session['git_commit']} | {self.session['gpu']}\n"
                        f"===== {self.command}\n")
        self.stdout = sys.stdout
        sys.stdout = _Tee(self.stdout, self.file)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self.stdout
        self.session["ended"] = now()
        self.session["status"] = "finished" if exc_type is None else f"stopped: {exc_type.__name__}: {exc}"
        save_sessions(self.run_dir, self.sessions)
        self.file.write(f"===== session ended {self.session['ended']} ({self.session['status']})\n")
        self.file.close()
        return False


def load_sessions(run_dir: Path) -> list:
    f = Path(run_dir) / "sessions.json"
    return json.load(open(f)) if f.exists() else []


def save_sessions(run_dir: Path, sessions: list):
    with open(Path(run_dir) / "sessions.json", "w") as f:
        json.dump(sessions, f, indent=2)


def _ci(m, key, metric="quadratic_kappa"):
    c = m.get(key, {}).get(metric)
    return f"[{c['lo']:.3f}, {c['hi']:.3f}]" if c else ""


def write_record(run_dir: Path):
    """RUN_RECORD.md, built only from the files of the run (metrics.json, history.json,
    sessions.json, the compare_*.json next to it). If a file is missing we say so."""
    run_dir = Path(run_dir)
    name = run_dir.name
    results = run_dir.parent
    if not run_dir.exists():
        print(f"[record] {run_dir} doesn't exist yet, nothing to write")
        return None
    L = [f"# Run record: {name}", "",
         f"Written automatically on {now()} by `python -m src.record`, from the files in this "
         f"folder. Nothing here is typed by hand.", ""]

    metrics_f, hist_f = run_dir / "metrics.json", run_dir / "history.json"
    m = json.load(open(metrics_f)) if metrics_f.exists() else None
    h = json.load(open(hist_f)) if hist_f.exists() else None

    L += ["## Status", ""]
    if m:
        L.append("Finished: training (if any) and evaluation are done, see `metrics.json`.")
    elif h:
        L.append(f"Not finished: {len(h)} epoch(s) trained so far, no test evaluation yet. "
                 "Run the same notebook again and it continues from the last saved epoch.")
    else:
        L.append("Nothing saved yet.")
    L.append("")

    L += ["## Sessions", ""]
    sessions = load_sessions(run_dir)
    if sessions:
        L += ["| started | ended | status | code version | GPU |", "|---|---|---|---|---|"]
        for s in sessions:
            # a session that was killed (laptop off, Colab disconnected) never wrote its end
            ended = s["ended"] or "never written"
            status = s["status"] if s["ended"] else "cut (Colab disconnected or the computer turned off)"
            L.append(f"| {s['started']} | {ended} | {status} | {s['git_commit']} | {s['gpu']} |")
        L += ["", "Command of the last session:", "", f"```\n{sessions[-1]['command']}\n```"]
    else:
        L.append("Not recorded (this run was done before sessions.json existed, Nov 3, 2026). "
                 "Its console output is kept in the repo under `docs/runs/`.")
    L.append("")

    if m:
        tc = m.get("train_config", {})
        dc = m.get("data_config", {})
        L += ["## Settings", "",
              f"* model: {m.get('model_name')}, seed: {m.get('seed')}",
              f"* crop: {dc.get('crop_method', 'n/a')}, imbalance: {dc.get('imbalance_strategy', 'n/a')}, "
              f"thicken strokes: {dc.get('thicken_strokes', 'n/a')}",
              f"* max epochs: {tc.get('epochs', 'n/a')}, patience: {tc.get('early_stopping_patience', 'n/a')}, "
              f"epoch kept by: {tc.get('select_metric', 'n/a')}",
              f"* checkpoint evaluated: `{Path(m.get('checkpoint', '')).name}`",
              f"* best epoch: {m.get('best_epoch')}", ""]
        a, t = m["argmax"], m["tuned"]
        L += ["## Test results", "",
              f"Test set: {m['n_test']:,} clocks. 95% intervals from resampling participants.", "",
              "| | kappa | accuracy | macro F1 |", "|---|---|---|---|",
              f"| argmax | {a['quadratic_kappa']:.3f} {_ci(m, 'argmax_ci')} | {a.get('accuracy', float('nan')):.3f} "
              f"| {a.get('macro_f1', float('nan')):.3f} |",
              f"| tuned cut points | {t['quadratic_kappa']:.3f} {_ci(m, 'tuned_ci')} | {t.get('accuracy', float('nan')):.3f} "
              f"| {t.get('macro_f1', float('nan')):.3f} |", ""]
        if m.get("tuned_thresholds"):
            L += [f"Cut points picked on validation: {', '.join(f'{x:.2f}' for x in m['tuned_thresholds'])}", ""]

    comps = sorted(results.glob(f"compare_{name}_vs_*.json")) + sorted(results.glob(f"compare_*_vs_{name}.json"))
    if comps:
        L += ["## Paired comparisons", "", "| A | B | kappa A - kappa B (95% CI) |", "|---|---|---|"]
        for f in comps:
            c = json.load(open(f))
            a_, b_ = f.stem[len("compare_"):].split("_vs_")
            L.append(f"| {a_} | {b_} | {c['diff']:+.3f} [{c['diff_lo']:+.3f}, {c['diff_hi']:+.3f}] |")
        L += ["", "If the interval doesn't cross 0, the difference is real (not noise).", ""]

    if h:
        L += ["## Every epoch", "", "| epoch | train loss | train acc | val loss | val acc | val kappa |",
              "|---|---|---|---|---|---|"]
        best = m.get("best_epoch") if m else None
        for e in h:
            star = " (kept)" if best is not None and e["epoch"] == best else ""
            L.append(f"| {e['epoch']}{star} | {e['train_loss']:.4f} | {e['train_acc']:.3f} | "
                     f"{e['val_loss']:.4f} | {e['val_acc']:.3f} | {e['val_kappa']:.3f} |")
        L.append("")

    L += ["## Files", "",
          "* `metrics.json`: every number above, as the run wrote it",
          "* `history.json`: every epoch",
          "* `sessions.json` and `train_log.txt`: when it ran and everything it printed",
          "* `predictions_test.csv`, `predictions_val.csv`: one row per clock (has participant IDs, "
          "keep it on Drive, never on GitHub)",
          "* `confusion_matrix_normalized.png`, `classification_report.json`", ""]

    out = run_dir / "RUN_RECORD.md"
    out.write_text("\n".join(L))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    for r in args.runs:
        out = write_record(args.results / r)
        if out:
            print(f"[record] wrote {out}")
