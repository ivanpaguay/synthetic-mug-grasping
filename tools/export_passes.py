# =========================================================================
# export_passes  v1   (training venv: numpy + Pillow)
# Colourised dataset passes for the reel and the report.
# It reads the BUILT dataset (pipeline/build_dataset.py), not the EXRs: the
# depth, instance, class and part PNGs hold the exact ids.
#
# For each selected image it writes, in --out/<scene>_v<V>_g<G>/ (all at the
# same size, ready for wipes and dissolves in Resolve):
#   rgb.png        the image as is
#   depth.png      depth with the "turbo" colour map (near = warm), black background
#   instance.png   one colour per instance (table dark grey, background black)
#   class.png      one colour per class (mug, distractors, gripper, table...)
#   part.png       mug parts (report colours) over the dimmed image
#   normal.png     camera-space normals (normal-map style: facing = blue), if present
#   strip.jpg      the 6 passes in a row, with captions (for the report)
#   grid.jpg       the 6 passes in 3x2, with captions
#
# Selection:
#   --images 10023:2:1,10500:0:0   scene:view:gripper (1 = with gripper)
#   (empty)  automatic: --n images spread across train scenes, alternating
#            with and without gripper, skipping heavily occluded ones
#
# Usage:
#   python export_passes.py --dataset <SYNTH_ROOT>/HIPS/dataset_E --out <SYNTH_ROOT>/reel/passes
# Options: --n 12, --seed 7 (instance colours)
# =========================================================================
import argparse
import colorsys
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PART_COLORS = {1: (150, 150, 150), 2: (230, 30, 30), 3: (30, 200, 30), 4: (230, 200, 30), 5: (40, 80, 230)}
CLASS_PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                 (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (127, 127, 127)]
INSTANCE_BG = 65535


def turbo(x):
    """Google's 'turbo' colour map as a polynomial; x in [0,1] -> uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.stack([r, g, b], -1).clip(0, 1) * 255).astype(np.uint8)


def depth_color(d16):
    valid = d16 > 0
    out = np.zeros(d16.shape + (3,), np.uint8)
    if valid.any():
        v = d16[valid].astype(np.float32)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        x = 1.0 - (d16.astype(np.float32) - lo) / max(hi - lo, 1.0)       # near = 1 (warm)
        out[valid] = turbo(x[valid])
    return out


def id_color(i, seed):
    h = ((i * 0.6180339887) + seed * 0.1234) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def instance_color(inst, seed):
    out = np.zeros(inst.shape + (3,), np.uint8)
    for i in np.unique(inst):
        if i == INSTANCE_BG:
            continue
        out[inst == i] = (60, 60, 60) if i == 0 else id_color(int(i), seed)
    return out


def class_color(cls):
    out = np.zeros(cls.shape + (3,), np.uint8)
    for c in np.unique(cls):
        if c == 0:
            continue
        out[cls == c] = CLASS_PALETTE[(int(c) - 1) % len(CLASS_PALETTE)]
    return out


def part_color(part, rgb):
    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)              # dimmed image as background
    for k, c in PART_COLORS.items():
        out[part == k] = c
    return out


def normal_color(npz_path, shape):
    if not os.path.isfile(npz_path):
        return None
    n = np.load(npz_path)["normal"].astype(np.float32)
    if n.shape[:2] != shape:
        return None
    valid = np.abs(n).sum(-1) > 0
    out = np.zeros(shape + (3,), np.uint8)
    m = n[valid] * np.array([1.0, -1.0, -1.0], np.float32)          # OpenCV -> standard view (facing = blue)
    out[valid] = ((m * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return out


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def labeled(tiles, labels, cols, tile_w):
    ims = []
    for t in tiles:
        im = Image.fromarray(t)
        ims.append(im.resize((tile_w, int(round(im.height * tile_w / im.width))), Image.LANCZOS))
    th = ims[0].height
    bar, gap = 34, 6
    rows = (len(ims) + cols - 1) // cols
    W = cols * tile_w + (cols - 1) * gap
    H = rows * (th + bar) + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    f = font(20)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        r, c = divmod(k, cols)
        x, y = c * (tile_w + gap), r * (th + bar + gap)
        d.text((x + 8, y + 7), lab, fill=(235, 235, 235), font=f)
        canvas.paste(im, (x, y + bar))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    with open(os.path.join(a.dataset, "index.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    key = lambda r: (int(r["scene_id"]), int(r["view_id"]), int(r["with_gripper"]))
    byk = {key(r): r for r in rows}
    if a.images:
        sel = []
        for tok in [t.strip() for t in a.images.split(",") if t.strip()]:
            s, v, g = (int(x) for x in tok.split(":"))
            if (s, v, g) in byk:
                sel.append(byk[(s, v, g)])
            else:
                print("[warning] not in the dataset: %s" % tok)
    else:
        cand = [r for r in rows if r.get("split") == "train"
                and str(r.get("target_heavily_occluded", "0")) in ("0", "", "False")]
        scenes = sorted({int(r["scene_id"]) for r in cand})
        pick = [scenes[int(i)] for i in np.linspace(0, len(scenes) - 1, min(a.n, len(scenes)))] if scenes else []
        sel = []
        for k, s in enumerate(pick):
            g = k % 2
            opts = sorted([r for r in cand if int(r["scene_id"]) == s and int(r["with_gripper"]) == g],
                          key=lambda r: int(r["view_id"]))
            opts = opts or [r for r in cand if int(r["scene_id"]) == s]
            if opts:
                sel.append(opts[k % len(opts)])
    os.makedirs(a.out, exist_ok=True)
    listing = []
    for r in sel:
        rgb_p = os.path.join(a.dataset, r["rgb"])
        base = lambda sub, ext="png": os.path.splitext(rgb_p.replace("/rgb/", "/%s/" % sub).replace("\\rgb\\", "\\%s\\" % sub))[0] + "." + ext
        rgb = np.array(Image.open(rgb_p).convert("RGB"))
        d16 = np.array(Image.open(base("depth"))).astype(np.uint16)
        inst = np.array(Image.open(base("instance"))).astype(np.int32)
        cls = np.array(Image.open(base("class"))).astype(np.int32)
        part = np.array(Image.open(base("part"))).astype(np.int32)
        nrm = normal_color(base("normal", "npz"), rgb.shape[:2])
        passes = [("rgb", "RGB", rgb), ("depth", "Depth", depth_color(d16)),
                  ("instance", "Instance", instance_color(inst, a.seed)), ("class", "Class", class_color(cls)),
                  ("part", "Mug parts", part_color(part, rgb))]
        if nrm is not None:
            passes.append(("normal", "Normals", nrm))
        name = "%06d_v%s_g%s" % (int(r["scene_id"]), r["view_id"], r["with_gripper"])
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        for fn, _, img in passes:
            Image.fromarray(img).save(os.path.join(d, fn + ".png"))
        tiles, labels = [p[2] for p in passes], [p[1] for p in passes]
        labeled(tiles, labels, len(tiles), 480).save(os.path.join(d, "strip.jpg"), quality=92)
        labeled(tiles, labels, 3, 640).save(os.path.join(d, "grid.jpg"), quality=92)
        listing.append(name)
        print("written: %s" % name)
    with open(os.path.join(a.out, "list.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(listing) + "\n")
    print("images: %d  ->  %s" % (len(listing), a.out))


if __name__ == "__main__":
    main()
