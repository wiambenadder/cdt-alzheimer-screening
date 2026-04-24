"""
Build a unified labels.csv from multiple NHATS rounds.

NHATS ships one SAS file per round (NHATS_Round_N_SP_File.sas7bdat) and
one folder of clock-drawing TIFFs per round. Each round's clock score
lives in a column named `cgNdclkdraw` (e.g. `cg1dclkdraw`, `cg11dclkdraw`).
The score is an ordinal 0..5 where:

    0 = Not recognizable as a clock
    1 = Severely distorted
    2 = Moderately distorted
    3 = Mildly distorted
    4 = Reasonably accurate
    5 = Accurate (circular or square)

Negative values (-1, -4, -7, -9) are NHATS missing-data codes and are
dropped.

This script:
  * iterates every round folder found under nhats_raw/
  * loads the matching SAS file from sas_files/
  * extracts (participant_id, cdt_score) per round
  * joins with TIFFs found on disk so we only keep images we actually have
  * writes a single labels.csv with columns:
        round, participant_id, cdt_score, image_path

It's safe to re-run — it will skip rounds with no SAS file and report
what's missing.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# --- Paths (override these from a notebook via build_labels(...)) ------------
DEFAULT_SAS_DIR = Path('/content/drive/MyDrive/cdt-data/sas_files')
DEFAULT_TIF_DIR = Path('/content/drive/MyDrive/cdt-data/nhats_raw')
DEFAULT_OUTPUT  = Path('/content/drive/MyDrive/cdt-data/labels.csv')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_sas_file(sas_dir: Path, round_num: int) -> Optional[Path]:
    """
    NHATS SAS filename patterns vary between rounds:
      NHATS_Round_1_SP_File.sas7bdat
      NHATS_Round_11_SP_File_V2.sas7bdat
    We match loosely: contains 'Round_<N>' and 'SP_File' and ends .sas7bdat.
    """
    pattern = re.compile(
        rf'NHATS_Round_{round_num}_SP_File.*\.sas7bdat$', re.IGNORECASE,
    )
    matches = [p for p in sas_dir.glob('*.sas7bdat') if pattern.search(p.name)]
    if not matches:
        return None
    # prefer the _V2 file if multiple versions exist
    matches.sort(key=lambda p: ('_V2' not in p.name, p.name))
    return matches[0]


def find_spid_column(columns) -> Optional[str]:
    for c in columns:
        if c.lower() == 'spid':
            return c
    return None


def _read_sas_robust(sas_path: Path) -> pd.DataFrame:
    """
    NHATS SAS files are inconsistently encoded across rounds. Some use
    UTF-8, others use Latin-1 / Windows-1252. Try a few common encodings
    before giving up.
    """
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return pd.read_sas(sas_path, format='sas7bdat', encoding=enc)
        except (UnicodeDecodeError, ValueError) as e:
            last_err = e
            continue
    # last-resort: let pandas return raw bytes for text columns; we only
    # need the numeric SPID and score columns so this still works
    print(f'  [warn] all encodings failed; loading as bytes ({last_err})')
    return pd.read_sas(sas_path, format='sas7bdat', encoding=None)


def extract_one_round(sas_path: Path, tif_dir: Path, round_num: int
                      ) -> pd.DataFrame:
    """
    Returns a dataframe with columns [round, participant_id, cdt_score, image_path]
    for a single round. Rows with missing scores or missing TIFFs are dropped.
    """
    score_col = f'cg{round_num}dclkdraw'
    print(f'[round {round_num:02d}] loading {sas_path.name}...')
    df = _read_sas_robust(sas_path)

    spid_col = find_spid_column(df.columns)
    if spid_col is None:
        print(f'  [skip] no SPID column in {sas_path.name}')
        return pd.DataFrame()
    if score_col not in df.columns:
        print(f'  [skip] column {score_col} not present. Columns like it: '
              f'{[c for c in df.columns if "clk" in c.lower()][:5]}')
        return pd.DataFrame()

    sub = df[[spid_col, score_col]].copy()
    sub.columns = ['participant_id', 'cdt_score']

    # drop NaNs and NHATS missing codes
    sub = sub.dropna(subset=['cdt_score'])
    sub = sub[sub['cdt_score'].between(0, 5)]
    sub['cdt_score'] = sub['cdt_score'].astype(int)

    # SPID often comes back as float from pandas.read_sas — clean to string
    sub['participant_id'] = sub['participant_id'].astype('Int64').astype(str)

    # match against TIFFs actually on disk
    tifs_on_disk = {p.stem: p for p in tif_dir.glob('*.tif')}
    tifs_on_disk.update({p.stem: p for p in tif_dir.glob('*.TIF')})  # just in case
    sub['image_path'] = sub['participant_id'].map(
        lambda pid: str(tifs_on_disk[pid]) if pid in tifs_on_disk else None
    )
    matched = sub['image_path'].notna().sum()
    sub = sub.dropna(subset=['image_path'])

    print(f'  [ok] {len(sub):,} labels matched to TIFFs '
          f'(of {matched:,} possible, out of {len(tifs_on_disk):,} TIFFs on disk)')

    sub.insert(0, 'round', round_num)
    return sub[['round', 'participant_id', 'cdt_score', 'image_path']]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_labels(
    sas_dir: Path = DEFAULT_SAS_DIR,
    tif_dir: Path = DEFAULT_TIF_DIR,
    output_csv: Path = DEFAULT_OUTPUT,
    rounds: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Walk every round subfolder under tif_dir, find its SAS file, extract
    and merge labels. Writes labels.csv and returns the combined df.

    If rounds is None, auto-detects by looking for folders named
    round_01, round_02, ...
    """
    sas_dir = Path(sas_dir)
    tif_dir = Path(tif_dir)

    if rounds is None:
        rounds = []
        for sub in sorted(tif_dir.iterdir()):
            m = re.match(r'round_(\d+)', sub.name)
            if sub.is_dir() and m:
                rounds.append(int(m.group(1)))
        if not rounds:
            print(f'[error] no round_NN subfolders under {tif_dir}')
            sys.exit(1)
        print(f'[auto-detect] found rounds: {rounds}')

    frames = []
    for r in rounds:
        sas_path = find_sas_file(sas_dir, r)
        if sas_path is None:
            print(f'[round {r:02d}] no SAS file found in {sas_dir}, skipping')
            continue
        this_tif_dir = tif_dir / f'round_{r:02d}'
        if not this_tif_dir.exists():
            print(f'[round {r:02d}] no TIFF folder at {this_tif_dir}, skipping')
            continue
        frames.append(extract_one_round(sas_path, this_tif_dir, r))

    if not frames:
        print('[error] no rounds produced any labels')
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    # --- Summary ---
    print('\n' + '=' * 60)
    print(f'[ok] wrote {len(combined):,} labels to {output_csv}')
    print('=' * 60)
    print('\nCounts per round:')
    print(combined.groupby('round').size().to_string())
    print('\nOverall class distribution:')
    print(combined['cdt_score'].value_counts().sort_index().to_string())
    print(f'\nUnique participants (some appear in >1 round): '
          f'{combined["participant_id"].nunique():,}')
    return combined


if __name__ == '__main__':
    build_labels()
