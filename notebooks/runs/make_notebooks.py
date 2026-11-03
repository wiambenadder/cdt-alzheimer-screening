"""
Writes the three seed notebooks of run B (B_vit_v2_seed42/43/44.ipynb) from one template,
so they are exactly the same except for the seed. If you change the template, run:

    python notebooks/runs/make_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLAB = "https://colab.research.google.com/github/wiambenadder/cdt-alzheimer-screening/blob/main/notebooks/runs"

STATUS = {
    42: "**Done** on Nov 3, 2026 (A100, 14 epochs, about 5.3 hours). Its console output is kept in "
        "`docs/runs/vit_b16_v2_s42.txt`. Running this notebook again does **not** retrain it: it "
        "only writes its `RUN_RECORD.md` on Drive from the files the run saved (a few minutes, no data copy).",
    43: "**Not done yet.** The first try (Nov 3) stopped when the laptop turned off, before this "
        "notebook could continue a cut run, so it starts from epoch 0.",
    44: "**Not done yet.**",
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(seed: int) -> dict:
    cells = [
        md(f"""
# Run B, seed {seed}: ViT-B/16 with the v2 preprocessing

This notebook trains the ViT **once, with seed {seed}**, on the same participant split as every other
run, then tests it on the same 8,938 test clocks, compares it with the summer ViT, and writes the
record of the run. One notebook per seed, so every seed has its own file and its own record.

**Status:** {STATUS[seed]}

**How long:** about 35 min to copy the clocks from Drive, about 4 min to check them, then about
5 to 6 hours of training (about 22 min per epoch on an A100, it stops by itself, 25 epochs at most).

**If Colab disconnects or the laptop sleeps:** open this same notebook again and do Runtime > Run all,
without changing anything. After every epoch the whole training state is saved on Drive, so it
continues from the last finished epoch. You lose at most the epoch that was running.

**Rules**
* Runtime > Change runtime type > A100 GPU (or L4)
* Never "Save a copy in GitHub" from Colab, the repo is public. Everything is saved on Drive.
* Don't change the seed here. Seed {seed} belongs to this notebook only.

**What it saves on Drive**, in `cdt-data/results_v2/vit_b16_v2_s{seed}/`:

| file | what's in it |
|---|---|
| `RUN_RECORD.md` | the readable record: sessions, settings, test results, every epoch (written from the files below, never by hand) |
| `metrics.json` | every test number with its 95% interval |
| `history.json` | every epoch (loss, accuracy, validation kappa), written after each epoch |
| `train_log.txt` | everything the run printed, from every session |
| `sessions.json` | each Colab session: start, end, code version, GPU, command |
| `predictions_test.csv` | one row per test clock (has participant IDs, stays on Drive) |

and the model in `cdt-data/models/vit_b16_v2_s{seed}_best.pt`.
"""),
        code("""
# we mount Drive (the NHATS data, the models and the results are there) and get the latest code
from google.colab import drive
drive.mount('/content/drive')

import os
REPO = '/content/cdt-alzheimer-screening'
if not os.path.exists(REPO):
    !git clone -q https://github.com/wiambenadder/cdt-alzheimer-screening.git {REPO}
else:
    !git -C {REPO} pull -q
%cd {REPO}
!pip install -q -r requirements.txt
!git log --oneline -1

import torch
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > A100 GPU, then Run all again'
print('GPU:', torch.cuda.get_device_name(0))
"""),
        code(f"""
SEED = {seed}                      # this notebook is only for this seed
RUN = f'vit_b16_v2_s{{SEED}}'
DRIVE_DATA = '/content/drive/MyDrive/cdt-data'
WORKERS = os.cpu_count()          # reading the TIFFs is the slow part, so we use every CPU

os.environ['CDT_DATA_DIR'] = DRIVE_DATA
os.environ['CDT_MODELS_DIR'] = f'{{DRIVE_DATA}}/models'
os.environ['CDT_RESULTS_DIR'] = f'{{DRIVE_DATA}}/results_v2'
RUN_DIR = f"{{os.environ['CDT_RESULTS_DIR']}}/{{RUN}}"

import json
DONE = os.path.exists(f'{{RUN_DIR}}/metrics.json')
epochs_done = len(json.load(open(f'{{RUN_DIR}}/history.json'))) if os.path.exists(f'{{RUN_DIR}}/history.json') else 0
can_continue = os.path.exists(f"{{os.environ['CDT_MODELS_DIR']}}/{{RUN}}_last.pt")
if DONE:
    print(f'{{RUN}} is already finished, this notebook will only write its record (no data copy, no training)')
elif can_continue:
    print(f'{{RUN}} was cut after {{epochs_done}} epoch(s), it will continue from there')
else:
    print(f'{{RUN}} starts from epoch 0')
"""),
        code("""
# we copy the TIFFs to the local disk first (reading 59k files from Drive during training is too
# slow), then we check every file: one that is missing or a different size than on Drive is copied
# again, then opened fully. A half-copied file crashed a training run once (Nov 3).
if DONE:
    print('already finished, no need to copy the clocks')
else:
    import shutil, time
    import pandas as pd
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    LOCAL = Path('/content/nhats_local')
    LOCAL.mkdir(exist_ok=True)
    RAW = Path(f'{DRIVE_DATA}/nhats_raw')
    t0 = time.time()
    for rf in sorted(RAW.iterdir()):
        if not rf.is_dir():
            continue
        dst = LOCAL / rf.name
        if dst.exists() and any(dst.glob('*.tif')):
            continue
        print(f'[copy] {rf.name}')
        shutil.copytree(rf, dst, dirs_exist_ok=True)
    print(f'staged in {(time.time() - t0) / 60:.1f} min')

    labels = pd.read_csv(f'{DRIVE_DATA}/labels.csv')
    labels['image_path'] = labels['image_path'].str.replace(f'{DRIVE_DATA}/nhats_raw/', str(LOCAL) + '/', regex=False)

    def check(path):
        local = Path(path)
        drive_file = RAW / local.relative_to(LOCAL)
        try:
            if not local.exists() or local.stat().st_size != drive_file.stat().st_size:
                shutil.copy2(drive_file, local)
            with Image.open(local) as im:
                im.load()
            return None
        except Exception:
            return path

    t0 = time.time()
    with ThreadPoolExecutor(16) as pool:
        bad = [p for p in pool.map(check, labels['image_path']) if p]
    print(f'checked {len(labels):,} files in {(time.time() - t0) / 60:.1f} min')
    # still broken = broken on Drive too, so we leave them out (the split is by participant,
    # so nobody moves between train, val and test). Only the count is printed, the names are IDs
    if bad:
        Path(f'{DRIVE_DATA}/broken_files.txt').write_text('\\n'.join(bad))
        labels = labels[~labels['image_path'].isin(bad)]
        print(f'{len(bad)} files are broken even on Drive and are left out (list in cdt-data/broken_files.txt)')
    else:
        print('all files open fine')
    LABELS = '/content/labels_local.csv'
    labels.to_csv(LABELS, index=False)
    print(f'{len(labels):,} labels -> {LABELS}')
"""),
        md("""
## Training

`--resume` continues a run that was cut, and a run that is already finished is skipped, so this
cell is always safe to run again. The lines to look for: `device=cuda`, then one `[train] ep=..`
line per epoch, then `test kappa (argmax)` and `test kappa (tuned)`.
"""),
        code("""
if not DONE:
    !python -m src.run --models vit_b16 --seeds {SEED} --select-metric val_kappa \\
        --run-tag _v2 --labels {LABELS} --num-workers {WORKERS} --resume
"""),
        md("""
## Comparison with the summer ViT and the record

The paired comparison uses the same test clocks for both models. If its interval doesn't cross 0,
the difference is real. Then the record is written from the saved files and shown below.
"""),
        code("""
if os.path.exists(f'{RUN_DIR}/predictions_test.csv'):
    !python -m src.run --compare {RUN} vit_b16_v1_s42
!python -m src.record {RUN}

from IPython.display import Markdown, display
record = f'{RUN_DIR}/RUN_RECORD.md'
display(Markdown(open(record).read()) if os.path.exists(record) else 'no record yet')
"""),
        md("""
## After this notebook

1. Paste the record above (or the `test kappa` and `[compare]` lines) in the chat.
2. Open `notebooks/06_results_report.ipynb` (CPU is fine) to put all the runs side by side.
"""),
    ]
    return {"nbformat": 4, "nbformat_minor": 0, "cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    for seed in (42, 43, 44):
        out = HERE / f"B_vit_v2_seed{seed}.ipynb"
        with open(out, "w") as f:
            json.dump(notebook(seed), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("wrote", out.name, "->", f"{COLAB}/{out.name}")
