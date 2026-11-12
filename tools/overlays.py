# =========================================================================
# overlays  v1   (training venv: torch + transformers + Pillow)
# Visual material for the reel and the report, from the FINAL TEST set.
# It changes no result: it only draws what the models already predict.
#
# For each selected photo it writes, in --out/<photo>/:
#   rgb.png                 the photo as is (full resolution)
#   mask_<model>.png        RGBA colour mask (alpha 0 on the background) for
#                           compositing in Resolve/Nuke: wipes, dissolves, etc.
#   over_<model>.png        photo + overlaid mask (alpha --alpha)
#   gt.png                  annotation in the same style (RGBA)
#   triptych.jpg            photo | model A | model B ... with captions
# and in --out:
#   per_photo.csv           mIoU / handle / mug IoU per photo and model (2 px band)
#   selection.txt           selected photos and why (best / typical / worst)
#
# Selection:
#   --photos 1501,1502,...  those photos
#   (empty)                 automatic, ranked by the LAST model in --models:
#                           the --n_best best, --n_mid around the median and
#                           the --n_worst worst (failures are shown too)
#
# Usage:
#   python overlays.py --real <SYNTH_ROOT>/real_photos/real_test_final
#       --models "Synthetic only=<SYNTH_ROOT>/train/runs/E_b2;Synthetic + 71 real=<SYNTH_ROOT>/train/runs/FT_b2_s0"
#       --out <SYNTH_ROOT>/reel/overlays
# Options: --ts path of train_seg.py (default: ../training/train_seg.py),
#   --alpha 0.55, --height 720 (height of each triptych panel),
#   --n_best 8 --n_mid 8 --n_worst 8
# Colours = the report's: body grey, handle red, rim green, interior yellow,
# base blue.
# =========================================================================
import argparse
import csv
import importlib.util
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {1: (150, 150, 150), 2: (230, 30, 30), 3: (30, 200, 30), 4: (230, 200, 30), 5: (40, 80, 230)}
NAMES = {1: "body", 2: "handle", 3: "rim", 4: "interior", 5: "base"}


def load_ts(path):
    spec = importlib.util.spec_from_file_location("train_seg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def slug(s):
    out = "".join(c.lower() if c.isalnum() else "_" for c in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def color_mask(lbl, alpha):
    h, w = lbl.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    for k, c in COLORS.items():
        m = lbl == k
        rgba[m, :3] = c
        rgba[m, 3] = int(round(255 * alpha))
    return rgba


def overlay(rgb, rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgb.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def legend_bar(width, fnt):
    h = 34
    img = Image.new("RGB", (width, h), (20, 20, 20))
    d = ImageDraw.Draw(img)
    x = 12
    for k in (1, 2, 3, 4, 5):
        d.rectangle([x, 10, x + 14, 24], fill=COLORS[k])
        d.text((x + 20, 8), NAMES[k], fill=(235, 235, 235), font=fnt)
        x += 30 + int(d.textlength(NAMES[k], font=fnt)) + 18
    return img


def triptych(panels, labels, height, fnt):
    tiles = []
    for p in panels:
        im = Image.fromarray(p)
        w = int(round(im.width * height / im.height))
        tiles.append(im.resize((w, height), Image.LANCZOS))
    bar = 40
    W = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
    canvas = Image.new("RGB", (W, height + bar), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    x = 0
    for t, lab in zip(tiles, labels):
        canvas.paste(t, (x, bar))
        d.text((x + 10, 9), lab, fill=(235, 235, 235), font=fnt)
        x += t.width + 8
    leg = legend_bar(W, font(16))
    out = Image.new("RGB", (W, canvas.height + leg.height), (20, 20, 20))
    out.paste(canvas, (0, 0))
    out.paste(leg, (0, canvas.height))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--photos", default="")
    ap.add_argument("--ts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "train_seg.py"))
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--n_best", type=int, default=8)
    ap.add_argument("--n_mid", type=int, default=8)
    ap.add_argument("--n_worst", type=int, default=8)
    a = ap.parse_args()

    models = []
    for part in [p for p in a.models.split(";") if p.strip()]:
        lab, path = part.split("=", 1)
        models.append((lab.strip(), path.strip()))
    if not models:
        raise SystemExit("--models is empty")

    ts = load_ts(a.ts)
    items = ts.real_items(a.real)
    byname = {it["name"]: it for it in items}
    os.makedirs(a.out, exist_ok=True)

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"

    # ---------- predictions of every model on every photo
    preds = {}          # (model, photo) -> label (np.uint8)
    scores = {it["name"]: {"photo": it["name"], "groups": " ".join(it["groups"])} for it in items}
    for lab, rd in models:
        with open(os.path.join(rd, "metrics.json"), encoding="utf-8") as fh:
            mname = json.load(fh)["args"]["model"]
        model = ts.build_model(mname).to(dev)
        model.load_state_dict(torch.load(os.path.join(rd, "best.pt"), map_location=dev))
        model.eval()
        with torch.no_grad():
            for it in items:
                x, y, _ = ts.EvalSet([it])[0]
                x, y = x.unsqueeze(0).to(dev), y.unsqueeze(0).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    lg = ts.logits_full(model, x)
                pred = lg.argmax(1)
                valid = y != 255
                band = ts.void_band(y.masked_fill(~valid, 0), max(1, round(a.band * y.shape[-1] / 1280.0)))
                s = ts.summarize(ts.confusion(pred, y, valid & ~band))
                k = slug(lab)
                scores[it["name"]][k + "_miou"] = s["miou_parts"]
                scores[it["name"]][k + "_handle"] = None if math.isnan(s["iou"]["handle"]) else s["iou"]["handle"]
                scores[it["name"]][k + "_mug"] = s["iou_mug"]
                preds[(lab, it["name"])] = pred[0].cpu().numpy().astype(np.uint8)
        print("model: %s (%s)" % (lab, mname))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    rows = list(scores.values())
    with open(os.path.join(a.out, "per_photo.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---------- selection
    why = {}
    if a.photos:
        sel = [p.strip() for p in a.photos.split(",") if p.strip()]
        for p in sel:
            why[p] = "picked by hand"
    else:
        k = slug(models[-1][0]) + "_miou"
        ranked = sorted([r for r in rows if r.get(k) is not None], key=lambda r: r[k], reverse=True)
        n = len(ranked)
        best = ranked[:a.n_best]
        worst = ranked[-a.n_worst:] if a.n_worst else []
        m0 = max(0, n // 2 - a.n_mid // 2)
        mid = ranked[m0:m0 + a.n_mid]
        sel = []
        for tag, grp in (("best", best), ("typical", mid), ("worst", worst)):
            for r in grp:
                if r["photo"] not in why:
                    why[r["photo"]] = "%s (%s mIoU %.3f)" % (tag, models[-1][0], r[k])
                    sel.append(r["photo"])
    missing = [p for p in sel if p not in byname]
    if missing:
        print("[warning] not in the set: %s" % ", ".join(missing))
    sel = [p for p in sel if p in byname]

    fnt = font(22)
    with open(os.path.join(a.out, "selection.txt"), "w", encoding="utf-8") as fh:
        for p in sel:
            fh.write("%s\t%s\t%s\n" % (p, why[p], scores[p]["groups"]))
    for p in sel:
        it = byname[p]
        d = os.path.join(a.out, p)
        os.makedirs(d, exist_ok=True)
        rgb = np.array(Image.open(it["rgb"]).convert("RGB"))
        Image.fromarray(rgb).save(os.path.join(d, "rgb.png"))
        gt = np.array(Image.open(it["part"]))
        if gt.shape != rgb.shape[:2]:
            gt = np.array(Image.fromarray(gt).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
        Image.fromarray(color_mask(gt, 1.0), "RGBA").save(os.path.join(d, "gt.png"))
        panels, labels = [rgb], ["Real photo"]
        for lab, _ in models:
            lbl = preds[(lab, p)]
            rgba = color_mask(lbl, a.alpha)
            k = slug(lab)
            Image.fromarray(color_mask(lbl, 1.0), "RGBA").save(os.path.join(d, "mask_%s.png" % k))
            ov = overlay(rgb, rgba)
            Image.fromarray(ov).save(os.path.join(d, "over_%s.png" % k))
            panels.append(ov)
            v = scores[p].get(k + "_miou")
            labels.append("%s  -  mIoU %.2f" % (lab, v) if v is not None else lab)
        triptych(panels, labels, a.height, fnt).save(os.path.join(d, "triptych.jpg"), quality=92)
    print("photos written: %d  ->  %s" % (len(sel), a.out))
    print("selection and reason: %s" % os.path.join(a.out, "selection.txt"))


if __name__ == "__main__":
    main()
