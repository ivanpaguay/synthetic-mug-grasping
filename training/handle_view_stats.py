# =========================================================================
# handle_view_stats  v1.1   (training venv: numpy + Pillow)
# Handle diagnostics: how many images show the handle FRONTALLY (in front of
# the body) and how many LATERALLY (against the background)?
#
# For each part mask with a visible handle (>= --min_px pixels) it looks at the
# 3 px ring around the handle and counts what is there:
#   frontal_score = mug pixels (body/rim/base/interior) / (mug + background)
#   frontal  if frontal_score >= 0.6   (the handle is surrounded by the body)
#   lateral  if frontal_score <= 0.3   (the handle sticks out against the background)
#   mixed    in between
#
# Usage:
#   python handle_view_stats.py --dataset <SYNTH_ROOT>/HIPS/dataset_E [--views 4,5]
#   python handle_view_stats.py --real <SYNTH_ROOT>/real_photos/real_test
# (--dataset: train images without gripper only; --real: folder with part/)
# =========================================================================
import argparse
import glob
import os

import numpy as np
from PIL import Image

HANDLE, IGN = 2, 255
MUG = (1, 3, 4, 5)


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def score(path, min_px):
    p = np.array(Image.open(path))
    h = p == HANDLE
    n = int(h.sum())
    if n < min_px:
        return None, n
    ring = dilate(h, 3) & ~h
    mug = np.isin(p[ring], MUG).sum()
    bg = (p[ring] == 0).sum()
    if mug + bg == 0:
        return None, n
    return mug / float(mug + bg), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--real", default="")
    ap.add_argument("--min_px", type=int, default=150)
    ap.add_argument("--views", default="")
    a = ap.parse_args()
    if a.dataset:
        import csv
        with open(os.path.join(a.dataset, "index.csv"), encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["split"] == "train" and r["with_gripper"] == "0"]
        if a.views:
            want = {v.strip() for v in a.views.split(",") if v.strip()}
            rows = [r for r in rows if r["view_id"] in want]
        files = [os.path.join(a.dataset, r["rgb"].replace("/rgb/", "/part/")) for r in rows]
        tag = "synthetic (train, no gripper%s)" % (", views " + a.views if a.views else "")
    else:
        files = sorted(glob.glob(os.path.join(a.real, "part", "*.png")))
        tag = "real"
    cnt = {"frontal": 0, "mixed": 0, "lateral": 0}
    px = {"frontal": [], "mixed": [], "lateral": []}
    nohandle = 0
    for f in files:
        if not os.path.isfile(f):
            continue
        s, n = score(f, a.min_px)
        if s is None:
            nohandle += 1
            continue
        k = "frontal" if s >= 0.6 else "lateral" if s <= 0.3 else "mixed"
        cnt[k] += 1
        px[k].append(n)
    tot = sum(cnt.values())
    print("%s: %d images, %d without a visible handle (< %d px)" % (tag, len(files), nohandle, a.min_px))
    for k in ("lateral", "mixed", "frontal"):
        med = int(np.median(px[k])) if px[k] else 0
        print("  %-8s %5d  (%5.1f %%)   handle px (median) %d" % (k, cnt[k], 100.0 * cnt[k] / max(1, tot), med))


if __name__ == "__main__":
    main()
