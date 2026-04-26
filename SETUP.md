# Setup Guide

This document provides step-by-step instructions for installing and running
the project. Estimated setup time: 10-15 minutes.

## 1. Prerequisites

- Python 3.10 or 3.11 (3.12 may also work; we developed on 3.10)
- `pip` or `conda`
- ~15 GB of free disk space for the NHATS images plus model checkpoints
- **Strongly recommended:** a CUDA-capable GPU (NVIDIA T4, L4, or better).
  Training ViT-B/16 for 25 epochs takes ~45 minutes on a T4, several hours on CPU.

## 2. Clone the repository

```bash
git clone https://github.com/wiambenadder/cdt-alzheimer-screening.git
cd cdt-alzheimer-screening
```

## 3. Install dependencies

### Option A: pip + venv (recommended)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate cdt-alz
```

Verify PyTorch sees the GPU:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 4. Obtain the NHATS dataset

NHATS data requires free registration and agreeing to a data-use agreement.

1. Visit https://nhats.org/researcher/data-access/public-use-files
2. Register for an account and accept the DUA
3. Download the Clock Drawing Test image files (rounds 1-9, 2011-2019)
4. Download the corresponding labels file (SAS/CSV) containing the CDT score

Place files as follows:

```
data/
  nhats_raw/
    round_01/
      <participant_id>.tif
    round_02/
      ...
    round_12/
      ...
  sas_files/
    NHATS_Round_1_SP_File.sas7bdat
    ...
  labels.csv
  sample_labels.csv
```

The labels CSV should have columns `participant_id` and `cdt_score` (0-5).
If NHATS ships different column names, update `src/config.py` and/or pass
the real column names to `load_labels()` in `src/data.py`.

> **Note to graders who cannot access NHATS:** See `data/README.md` for a
> small synthetic sample that verifies the pipeline runs end-to-end without
> requiring the real data.

## 5. Verify the pipeline

Run the smoke test to confirm everything is wired up:

```bash
python -m src.verify
```

This loads the labels CSV, checks that images are readable, builds a model,
and runs one forward pass. It should print `[ok] all checks passed`.

## 6. Reproduce the experiments

The main experiments are organized as notebooks. Run them in order:

```bash
jupyter lab
```

- `notebooks/01_data_exploration.ipynb` - inspect label distribution, sample images
- `notebooks/02_baseline.ipynb` - majority-class baseline + VGG16 frozen feature extractor
- `notebooks/03_finetuning.ipynb` - train VGG16, EfficientNet-B0, and ViT-B/16
- `notebooks/04_ablation.ipynb` - 2x2 ablation: {frozen, unfrozen} x {no-aug, aug}
- `notebooks/05_error_analysis.ipynb` - metrics, confusion matrices, Grad-CAM, error analysis

Each notebook writes its outputs to `docs/results/<run_name>/` and model
checkpoints to `models/<run_name>_best.pt`.

## 7. Train from the command line (optional)

If you prefer scripts over notebooks:

```bash
# train a single run
python -m src.train_cli --model vit_b16 --epochs 25 --run_name vit_b16_ft

# reproduce the full architecture-comparison sweep
python -m src.run_comparison

# run the 2x2 ablation
python -m src.run_ablation

# evaluate all trained runs on the test set and write metrics
python -m src.run_eval
```

## 8. External APIs / services

This project uses only publicly-available pretrained weights shipped with
`torchvision`. **No API keys required.** The first run of each model will
download ImageNet-pretrained weights (~100 MB for VGG16, ~20 MB for
EfficientNet-B0, ~340 MB for ViT-B/16) into `~/.cache/torch/hub/checkpoints/`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | lower `batch_size` in `config.py` (try 16 or 8) |
| `.tif` images fail to open | `pip install pillow` is up to date; some TIFFs need `libtiff` |
| Slow DataLoader | lower `num_workers` to 0 on Windows/macOS, raise to 8 on Linux |
| Training diverges (NaN loss) | lower `lr_head`; check that class weights aren't inf (happens if a class has 0 samples in train split) |
