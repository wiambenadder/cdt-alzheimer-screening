# Data

This folder is for the NHATS Clock Drawing Test data used in this project.

The real NHATS images and label files are **not stored in this GitHub repo**. They are ignored with `.gitignore` because:

1. NHATS data is shared under a data-use agreement.
2. The full image dataset is very large and would make the repo too big.

## The layout

```text
data/
  nhats_raw/
    round_01/
      40016109.tif
      ...
    round_02/
      ...
    ...
    round_14/
      ...
  sas_files/
    NHATS_Round_14_SP_File.sas7bdat
    ...
  labels.csv
  sample_labels.csv
```

## What each part contains

- `nhats_raw/` includes the raw clock drawing images.
- Each `round_XX/` folder consists of `.tif` files for that NHATS round.
- Image files are named with the participant ID, for example `40016109.tif`.
- `sas_files/` contains the NHATS SAS data files, such as `NHATS_Round_14_SP_File.sas7bdat`.
- `labels.csv` is the processed label file used by the training pipeline.
- `sample_labels.csv` can be used for quick smoke tests without the full dataset.

## How I organized the data

I kept the raw NHATS images in Google Drive inside `nhats_raw/`, grouped by round.
I kept the original NHATS SAS files in a separate `sas_files/` folder.
I then created a cleaned `labels.csv` file that links each participant ID to a CDT score and image path for training.

## How to get the data

1. Request access to NHATS data from the official NHATS researcher data portal:
   [[[https://nhats.org/researcher/data-access/public-use-files](https://nhats.org/researcher/data-access/public-use-files)](https://nhats.org/welcome)]
2. Download the clock drawing image files for the rounds you want to use.
3. Download the matching NHATS SAS/SP files that contain the variables to build labels.
4. Place the image folders inside `data/nhats_raw/`.
5. Place the SAS files inside `data/sas_files/`.
6. Build or copy `labels.csv` so it includes the columns used by this project.

## Labels used in this project

The main label file used by the pipeline is `labels.csv`.

It should contain these columns:

- `participant_id`
- `cdt_score`
- `image_path`

In this project, the model predicts CDT score from 0 to 5.

## Privacy and ethics

NHATS data should not be uploaded to a public repository.
Please keep the raw images, SAS files, and any derived label files in private storage such as Google Drive or local disk only.

