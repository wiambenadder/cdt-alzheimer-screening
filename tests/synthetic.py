"""
Fake NHATS-like clock pages, so we can test the whole pipeline without the real
data (which can't leave private storage under the data use agreement).

Each page looks roughly like a real scan:
  * a big white page (2560 x 3312, like round 1) in black and white
  * a small hand-drawn-ish clock somewhere on it, more broken for lower scores
  * scanner registration bars near the edges, and sometimes one further inside
  * sometimes a note written far from the clock, like the interviewer's "32:09"

We use:  python -m tests.synthetic <out_dir> [n_participants]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PAGE_W, PAGE_H = 2560, 3312


def draw_clock(draw, rng, cx, cy, r, score):
    """we draw a clock whose quality depends on the score (5 = accurate)."""
    stroke = 4
    # the outline gets more wobbly for lower scores
    wobble = {5: 0.02, 4: 0.05, 3: 0.08, 2: 0.12, 1: 0.2, 0: 0.3}[score]
    angles = np.linspace(0, 2 * np.pi, 120)
    radii = r * (1 + wobble * np.sin(angles * rng.integers(2, 5) + rng.random() * 6))
    pts = [(cx + rr * np.cos(a), cy + rr * np.sin(a)) for rr, a in zip(radii, angles)]
    if score >= 1:
        draw.line(pts + [pts[0]], fill=0, width=stroke)
    else:
        # score 0: just a broken arc
        draw.line(pts[:40], fill=0, width=stroke)

    # the numbers: all 12 for good clocks, fewer and misplaced for bad ones
    n_numbers = {5: 12, 4: 12, 3: 9, 2: 6, 1: 3, 0: 2}[score]
    for k in range(n_numbers):
        a = -np.pi / 2 + 2 * np.pi * (k + 1) / 12
        if score <= 2:
            a = -np.pi / 2 + rng.random() * np.pi * (0.8 if score == 2 else 0.4)
        rr = r * (0.8 if score >= 3 else rng.uniform(0.6, 1.3))
        x, y = cx + rr * np.cos(a), cy + rr * np.sin(a)
        # a crude digit made of strokes, the font size is too small to read anyway
        h = r * 0.12
        draw.line([(x, y - h), (x, y + h)], fill=0, width=stroke)
        if (k % 2) == 0:
            draw.line([(x, y - h), (x + h * 0.6, y - h)], fill=0, width=stroke)

    # the hands, only for scores 2 and up
    if score >= 2:
        for a, length in ((-np.pi / 2 - np.pi / 6, 0.5), (-np.pi / 6, 0.75)):
            a = a + (0 if score >= 4 else rng.normal(0, 0.6))
            draw.line([(cx, cy), (cx + r * length * np.cos(a), cy + r * length * np.sin(a))],
                      fill=0, width=stroke)


def make_page(rng, score, add_note=False, inner_bar=False):
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(img)

    # scanner bars near the edges (outside the 12% strip in most cases)
    draw.rectangle([150, PAGE_H - 250, 330, PAGE_H - 200], fill=0)
    draw.rectangle([PAGE_W - 400, 120, PAGE_W - 220, 170], fill=0)
    if inner_bar:
        # a bar inside the inner region, the kind that got into the v1 crop
        draw.rectangle([450, PAGE_H - 900, 520, PAGE_H - 760], fill=0)

    r = int(rng.uniform(250, 450))
    cx = int(rng.uniform(0.3, 0.6) * PAGE_W)
    cy = int(rng.uniform(0.25, 0.45) * PAGE_H)
    draw_clock(draw, rng, cx, cy, r, score)

    if add_note:
        # a short note far from the clock
        x0, y0 = int(0.6 * PAGE_W), int(0.7 * PAGE_H)
        for k in range(4):
            draw.line([(x0 + 60 * k, y0), (x0 + 60 * k + 30, y0 + 70)], fill=0, width=4)

    # real scans are 1-bit, so we save them like that
    return img.convert("1"), (cx, cy, r)


def make_dataset(out_dir, n_participants=60, n_rounds=3, seed=0):
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        pid = str(10000000 + p)
        for rnd in range(1, n_rounds + 1):
            if rng.random() < 0.2:
                continue   # not everybody draws in every round
            score = int(rng.choice(6, p=[0.02, 0.05, 0.12, 0.22, 0.34, 0.25]))
            page, _ = make_page(rng, score, add_note=rng.random() < 0.3,
                                inner_bar=rng.random() < 0.3)
            path = out_dir / "nhats_raw" / f"round_{rnd:02d}" / f"{pid}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            page.save(path)
            rows.append({"round": rnd, "participant_id": pid, "cdt_score": score,
                         "image_path": str(path)})
    labels = pd.DataFrame(rows)
    labels.to_csv(out_dir / "labels.csv", index=False)
    print(f"[synthetic] wrote {len(labels)} pages for {n_participants} participants to {out_dir}")
    return labels


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data_synthetic"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    make_dataset(out, n)
