# Data

This directory holds the NHATS Clock Drawing Test data. The actual images and
labels are **not committed to git** (they are gitignored in `.gitignore`) both
because NHATS data requires a data-use agreement and because ~40K TIFF images
would blow up the repo.

## Expected layout

```
data/
  nhats_raw/
    <participant_id_1>.tif
    <participant_id_2>.tif
    ...
  labels.csv                    # participant_id, cdt_score (0-5)
  sample_labels.csv             # tiny synthetic sample for smoke testing
```

## How to obtain NHATS CDT data

1. Go to https://nhats.org/researcher/data-access/public-use-files
2. Create an account and accept the data-use agreement (free, same-day)
3. Download CDT image files for the rounds you want (rounds 1-9 span 2011-2019)
4. Download the corresponding Sensitive Demographic File / questionnaire files
   containing the CDT score variable.

## Label column names

The NHATS variable naming convention is round-specific. Typical names you may
encounter include:

- `HC1DISECG` - clock drawing executive composite (older rounds)
- `CG1DCLOCKSCORE` - clock drawing score
- similar round-2..9 variants

Inspect your downloaded CSV in the `01_data_exploration` notebook and update
`src/config.py` or pass the real column name to `load_labels()`.

## Privacy and ethics

Clock drawings are de-identified in NHATS releases - only a participant ID
links images to scores. Do not commit the raw data to a public git repo.
