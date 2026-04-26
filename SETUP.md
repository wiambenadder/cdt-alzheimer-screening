# Setup Guide

This document provides step-by-step instructions for installing and running
the project. Estimated setup time: 10 to 15 minutes.

## 1. Prerequisites

- Python 3.10 or 3.11 (3.12 may also work; development was done on 3.10)
- pip or conda
- Around 15 GB of free disk space for the NHATS images and model checkpoints
- Strongly recommended: a CUDA-capable GPU (NVIDIA T4, L4, or better).
  Training ViT-B/16 for 25 epochs takes around 45 minutes on a T4 and
  several hours on CPU.

## 2. Clone the repository

```bash
git clone https://github.com/wiambenadder/cdt-alzheimer-screening.git
cd cdt-alzheimer-screening
```

## 3. Install dependencies

### Option A: pip and venv (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

### Option B: conda

```bash
conda env create -f environment.yml
conda activate cdt-alz
```

Verify PyTorch can see the GPU:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## 4. Obtain the NHATS dataset

NHATS data requires free registration and a data use agreement.

1. Visit https://nhats.org/researcher/data-access/public-use-files
2. Register for an account and accept the data use agreement
3. Download the Clock Drawing Test image files for rounds 1 through 12
4. Download the NHATS SP questionnaire SAS files for each of those rounds
   (these are the source of the CDT score labels)

Note: Rounds 13 and 14 changed the CDT variable structure in the SAS files,
so they cannot be used with this pipeline and should be skipped.

Place files as follows:
