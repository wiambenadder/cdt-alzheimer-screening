# Clock component v2

This file keeps track of what changed between the summer version (v1) of the
clock-drawing component and v2, and which runs we still need to do on Colab.
The v1 numbers in the README stay there until the v2 runs replace them.

## What was wrong in v1

| Problem | Where | Effect |
|---|---|---|
| The attribution maps were drawn on the full uncropped page, but computed on the cropped input | NB05 | The heatmaps and the pictures underneath did not line up, so the figure could not support "the model looks at the clock face and hands" |
| The occlusion patch was ImageNet mean gray | `gradcam.py` | On white paper a gray square looks like ink, which adds signal instead of removing it |
| `src/train.py` did not run as committed (3-space indent at line 160) and still used `torch.cuda.amp` | `train.py` | The repo could not reproduce its own results |
| The notebooks overwrote `src/` with `%%writefile` | NB01, 03, 04, 05 | The code in `src/` was not guaranteed to be the code that ran |
| NB02 and `verify.py` used the random-stratified split | NB02, `verify.py` | The frozen VGG16 baseline had participant leakage, so it was not comparable to the other runs |
| Rotation and affine augmentation filled the corners with black | `augmentation.py` | Every augmented image had black triangles that never happen on a real scan, which might be part of why augmentation hurt the frozen model |
| The crop could keep scanner bars and could come out non-square (2713x3160 in NB01) | `SmartCropClock` | Resize then stretched the clock |
| Class names said "severe impairment" ... "normal" | `config.py`, README | The NHATS score rates how well the clock is drawn, it is not a diagnosis |
| One seed per run and no confidence intervals | everywhere | 0.812 vs 0.808 vs the paper's 0.81 is within noise, and we had no way to tell |
| VGG16 was stopped by hand after 8 epochs | NB03 | The architecture comparison was not on equal terms |

## What v2 changes

* `src/` is the only copy of the code. `python -m src.run` trains and evaluates any configuration, and the notebooks just call into `src/`.
* `CDT_DATA_DIR`, `CDT_MODELS_DIR`, `CDT_RESULTS_DIR` move the folders to Drive without editing code.
* `get_splits()` saves the participant split to `data/splits.csv` the first time and reuses it after. It uses the same `GroupShuffleSplit` and seed as v1, so it produces the same test participants as long as `labels.csv` has not changed.
* `SmartCropClock(method="component")` removes solid rectangles (scanner bars), keeps the biggest ink blob plus nearby fragments, leaves far-away notes out, and pads to a true square with white. `method="projection"` is the v1 crop, untouched, for re-scoring the old checkpoints.
* Rotation and affine fill with white.
* Occlusion uses paper-white patches and the softmax probability, runs in batches, and `plot_attribution_grid()` draws the heatmap on the exact model input.
* `bootstrap_metrics()` gives 95% CIs by resampling participants (one person's clocks from different rounds are not independent). `paired_bootstrap_kappa()` tests whether one model really beats another.
* `optimize_thresholds()` turns the 6 probabilities into an expected score and picks the cut points on validation. This works on the existing checkpoints with no retraining.
* `train()` takes a `seed`, logs validation kappa, and can pick the best epoch by kappa (`--select-metric val_kappa`) instead of loss.
* Options for the next experiments: `--thicken-strokes`, `vit_b16_384`, `--imbalance weighted_sampler`.
* `tests/` has a synthetic page generator and tests that run on a laptop without the NHATS data.

## Runs to do on Colab

Before any of these, in the first cell:

```
%env CDT_DATA_DIR=/content/drive/MyDrive/cdt-data
%env CDT_MODELS_DIR=/content/drive/MyDrive/cdt-data/models
```

### A. Re-score the v1 checkpoints (no training, about 30 min)

This gives the v1 numbers with confidence intervals and the tuned cut points.
The v1 models were trained with the projection crop, so we evaluate them with it.

```
python -m src.run --models vit_b16 --evaluate-only --crop-method projection \
    --checkpoint $CDT_MODELS_DIR/vit_b16_ft_best.pt --run-tag _v1
python -m src.run --models vgg16 --evaluate-only --crop-method projection \
    --checkpoint $CDT_MODELS_DIR/vgg16_ft_best.pt --run-tag _v1
python -m src.run --models efficientnet_b0 --evaluate-only --crop-method projection \
    --checkpoint $CDT_MODELS_DIR/effb0_ft_best.pt --run-tag _v1
python -m src.run --compare vit_b16_v1_s42 vgg16_v1_s42
```

Then re-run the attribution cell in NB05 to get maps that line up.

### B. Retrain the main model with v2 preprocessing (3 seeds)

```
python -m src.run --models vit_b16 --seeds 42 43 44 --select-metric val_kappa --run-tag _v2
python -m src.run --compare vit_b16_v2_s42 vit_b16_v1_s42
```

### C. The new questions, one change at a time against B

| Run | Command change | Question |
|---|---|---|
| C1 | `--thicken-strokes` | Do thin strokes and small numbers get lost in the resize? |
| C2 | `--models vit_b16_384` | Does more resolution help? (slower, about 3x) |
| C3 | `--imbalance weighted_sampler` | Is oversampling better than a weighted loss? |
| C4 | `--models vgg16 efficientnet_b0` with the same settings as B | A fair architecture comparison (no manual stop) |

Each ViT run is a few hours on a T4. An L4 or A100 cuts that a lot.
