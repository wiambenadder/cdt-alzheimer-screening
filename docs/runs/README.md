# Runs of the clock model (v2)

This folder keeps track of every run we did on Colab, so every number in the report can be
traced back to where it came from. Nothing in the results is typed by hand: the numbers come
either from the files the runs write on Drive, or from the Colab output copied here as it was.

## Where the numbers come from

| What | Where it is | Who writes it |
|---|---|---|
| The Colab output of the runs before Nov 4, 2026 | `docs/runs/*.txt` (this folder) | copied from the notebook output without changes, only the TensorFlow startup warnings were removed |
| Every run, from Nov 4 on | Drive: `cdt-data/results_v2/<run>/RUN_RECORD.md` | `src/record.py`, from the files of the run |
| The scores of a run | Drive: `cdt-data/results_v2/<run>/metrics.json` | `src/run.py`, at the end of the run |
| Every epoch of a run | Drive: `cdt-data/results_v2/<run>/history.json` | `src/train.py`, after every epoch |
| Everything a run printed | Drive: `cdt-data/results_v2/<run>/train_log.txt` | `src/record.py`, while it runs |
| All the runs side by side | Drive: `cdt-data/results_v2/report/` (RESULTS.md, tables, figures) | `notebooks/06_results_report.ipynb` |

The files on Drive stay on Drive: the predictions have participant IDs and the NHATS data use
agreement doesn't allow them on GitHub. Only aggregate numbers (scores, curves) come here.

## The runs

| Run | What it is | Notebook | Status |
|---|---|---|---|
| Section A | the summer (v1) checkpoints scored again, with 95% intervals and tuned cut points | `notebooks/00_run_on_colab.ipynb` | done Nov 2, 2026, output in `section_A_v1_rescore.txt` |
| A2 | attribution maps of the v1 ViT, drawn on the model input | `notebooks/00_run_on_colab.ipynb` | done Nov 3, 2026, figure on Drive (`results_v2/vit_b16_v1_s42/attribution_maps.png`) |
| B, seed 42 | ViT-B/16 retrained with the v2 preprocessing | `notebooks/runs/B_vit_v2_seed42.ipynb` | done Nov 3, 2026, output in `vit_b16_v2_s42.txt` |
| B, seed 43 | same, seed 43 | `notebooks/runs/B_vit_v2_seed43.ipynb` | to do (the Nov 3 try stopped when the laptop turned off) |
| B, seed 44 | same, seed 44 | `notebooks/runs/B_vit_v2_seed44.ipynb` | to do |

Open in Colab:
[seed 42](https://colab.research.google.com/github/wiambenadder/cdt-alzheimer-screening/blob/main/notebooks/runs/B_vit_v2_seed42.ipynb) ·
[seed 43](https://colab.research.google.com/github/wiambenadder/cdt-alzheimer-screening/blob/main/notebooks/runs/B_vit_v2_seed43.ipynb) ·
[seed 44](https://colab.research.google.com/github/wiambenadder/cdt-alzheimer-screening/blob/main/notebooks/runs/B_vit_v2_seed44.ipynb) ·
[results side by side](https://colab.research.google.com/github/wiambenadder/cdt-alzheimer-screening/blob/main/notebooks/06_results_report.ipynb)

## Results so far

From `section_A_v1_rescore.txt` and `vit_b16_v2_s42.txt`. Same 8,938 test clocks (2,047
participants) for every run, quadratic weighted kappa with a 95% interval from resampling
participants. "Tuned" uses cut points picked on validation.

| Run | Kappa | Tuned kappa | Accuracy | Macro F1 |
|---|---|---|---|---|
| EfficientNet-B0 v1 | 0.747 [0.732, 0.763] | 0.782 [0.769, 0.795] | 0.590 | 0.550 |
| VGG16 v1 | 0.781 [0.767, 0.794] | 0.803 [0.790, 0.814] | 0.616 | 0.584 |
| ViT-B/16 v1 | 0.808 [0.795, 0.821] | 0.836 [0.825, 0.846] | 0.672 | 0.648 |
| ViT-B/16 v2, seed 42 | 0.834 [0.823, 0.845] | 0.849 [0.839, 0.858] | not printed | not printed |

Paired comparisons:
* ViT v1 vs VGG16 v1: +0.027 [+0.019, +0.035]
* ViT v2 seed 42 vs ViT v1: +0.026 [+0.019, +0.034]

The accuracy and macro F1 of seed 42 are in its `metrics.json` on Drive, they weren't printed in
the Colab output, so they aren't written here. Running `B_vit_v2_seed42.ipynb` (a few minutes)
writes its `RUN_RECORD.md` with them.

What this does and doesn't show:
* one seed of v2 only, seeds 43 and 44 tell if 0.834 is typical
* v2 changes three things at once (the crop, the white fill, the epoch picked by validation kappa
  instead of loss), so +0.026 is their effect together, not the crop alone
* tuned v1 (0.836) is about as good as plain v2 (0.834), the tuned vs tuned difference hasn't
  been tested yet

## If a run gets cut

Since Nov 4, 2026 the training state is saved on Drive after every epoch
(`cdt-data/models/<run>_last.pt`, about 1 GB, deleted when the run is finished). Opening the same
notebook again and doing Run all continues from the last finished epoch, so a disconnect costs at
most one epoch (about 22 min). A finished run is never trained again by accident: `src/run.py`
skips any run that already has `metrics.json` (`--force` to redo it on purpose).
A cut run that continues gives the same numbers as one that was never cut, the random number
generators are saved too (checked in `tests/test_pipeline.py`).

## Rules we follow

* one notebook per seed, the seed is written in the notebook and never changed
* never "Save a copy in GitHub" from Colab after running, the outputs would go on the public repo
* the numbers in the report come from `RUN_RECORD.md`, `RESULTS.md` or the files in this folder,
  never from memory
