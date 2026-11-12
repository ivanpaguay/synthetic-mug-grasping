# =========================================================================
# real_diag  v1   (training venv: torch + transformers + Pillow)
# Diagnostics on the real DEVELOPMENT set (never on the final test set):
# on which kind of photo does each model win or lose?
#
# For each real photo it computes, from its annotated mask:
#   handle:   frontal / mixed / lateral / no_handle  (same rule as handle_view_stats.py)
#   interior: low / mid / high = fraction of interior pixels over the mug, in
#             terciles of the real set itself. A proxy for camera ELEVATION
#             (grazing view -> little interior; top view -> a lot). It also
#             depends on the mug shape.
# and evaluates each model (best.pt in each --runs folder) per group:
#   parts mIoU, handle IoU and mug IoU (with a --band px band), and where the
#   annotated handle pixels go (background, body, handle, ...).
#
# With --dataset it adds the distribution of the interior proxy in the
# synthetic set (train, no gripper), split into views 0-3 and views 4-5.
#
# Usage:
#   python real_diag.py --real <SYNTH_ROOT>/real_photos/real_test
#       --runs runs/E_b0,runs/E2_b0_s0 --dataset <SYNTH_ROOT>/HIPS/dataset_E2
#       --csv runs/real_diag.csv
# Options: --ts path of train_seg.py (default: next to this script), --band 2
# =========================================================================
import argparse
import csv
import importlib.util
import json
import math
import os

import numpy as np
from PIL import Image

HANDLE, INTERIOR, IGN = 2, 4, 255
MUG = (1, 3, 4, 5)              # body, rim, interior, base (for the ring around the handle)
ALL_MUG = (1, 2, 3, 4, 5)


def load_ts(path):
    spec = importlib.util.spec_from_file_location("train_seg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def handle_kind(p, min_px):
    h = p == HANDLE
    if int(h.sum()) < min_px:
        return "no_handle"
    ring = dilate(h, 3) & ~h
    mug = np.isin(p[ring], MUG).sum()
    bg = (p[ring] == 0).sum()
    if mug + bg == 0:
        return "no_handle"
    s = mug / float(mug + bg)
    return "frontal" if s >= 0.6 else "lateral" if s <= 0.3 else "mixed"


def interior_frac(p):
    mug = np.isin(p, ALL_MUG).sum()
    return float((p == INTERIOR).sum()) / mug if mug else float("nan")


def bin_of(v, t1, t2):
    if math.isnan(v):
        return "interior_unknown"
    return "interior_low" if v < t1 else "interior_mid" if v < t2 else "interior_high"


def fmt(v):
    return "  -  " if v is None or (isinstance(v, float) and math.isnan(v)) else "%.3f" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--ts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_seg.py"))
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--min_px", type=int, default=150)
    ap.add_argument("--csv", default="real_diag.csv")
    a = ap.parse_args()

    ts = load_ts(a.ts)
    items = ts.real_items(a.real)

    # ---------- descriptors of each real photo (from its annotation)
    desc = []
    for it in items:
        p = np.array(Image.open(it["part"]))
        sc = (p.shape[1] / 1280.0) ** 2
        desc.append({"handle": handle_kind(p, a.min_px * sc), "intf": interior_frac(p)})
    iv = np.array([d["intf"] for d in desc if not math.isnan(d["intf"])])
    t1, t2 = (float(np.quantile(iv, 1 / 3.0)), float(np.quantile(iv, 2 / 3.0))) if len(iv) else (0.1, 0.2)
    for d in desc:
        d["int_bin"] = bin_of(d["intf"], t1, t2)
    groups = ["all", "frontal", "mixed", "lateral", "no_handle", "interior_low", "interior_mid", "interior_high"]
    cnt = {g: 0 for g in groups}
    for d in desc:
        for g in ("all", d["handle"], d["int_bin"]):
            if g in cnt:
                cnt[g] += 1
    print("real: %d photos | interior terciles: < %.3f low, < %.3f mid, rest high" % (len(items), t1, t2))
    print("  " + "  ".join("%s %d" % (g, cnt[g]) for g in groups[1:]))

    # ---------- interior proxy in the synthetic set
    if a.dataset:
        with open(os.path.join(a.dataset, "index.csv"), encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["split"] == "train" and r["with_gripper"] == "0"]
        for tag, want in (("views 0-3", {"0", "1", "2", "3"}), ("views 4-5", {"4", "5"})):
            vals = []
            for r in rows:
                if r["view_id"] not in want:
                    continue
                pp = os.path.join(a.dataset, r["rgb"].replace("/rgb/", "/part/"))
                if os.path.isfile(pp):
                    vals.append(interior_frac(np.array(Image.open(pp))))
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                continue
            b = [bin_of(v, t1, t2) for v in vals]
            n = float(len(b))
            print("synthetic %s: %d images | interior low %.0f %%  mid %.0f %%  high %.0f %%  (real: 33/33/33)" %
                  (tag, len(b), 100 * b.count("interior_low") / n, 100 * b.count("interior_mid") / n,
                   100 * b.count("interior_high") / n))

    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    if not runs:
        return

    # ---------- evaluation per model
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    NC = ts.NC
    per_img = [{"photo": it["name"], "handle_kind": d["handle"], "interior_frac": round(d["intf"], 4),
                "interior_bin": d["int_bin"]} for it, d in zip(items, desc)]
    table = {}
    for rd in runs:
        name = os.path.basename(os.path.normpath(rd))
        with open(os.path.join(rd, "metrics.json"), encoding="utf-8") as fh:
            mname = json.load(fh)["args"]["model"]
        model = ts.build_model(mname).to(dev)
        model.load_state_dict(torch.load(os.path.join(rd, "best.pt"), map_location=dev))
        model.eval()
        cms = {g: torch.zeros(NC, NC, dtype=torch.float64) for g in groups}
        with torch.no_grad():
            for k, it in enumerate(items):
                x, y, _ = ts.EvalSet([it])[0]
                x, y = x.unsqueeze(0).to(dev), y.unsqueeze(0).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    lg = ts.logits_full(model, x)
                pred = lg.argmax(1)
                valid = y != IGN
                band = ts.void_band(y.masked_fill(~valid, 0), max(1, round(a.band * y.shape[-1] / 1280.0)))
                cm = ts.confusion(pred, y, valid & ~band)
                d = desc[k]
                for g in ("all", d["handle"], d["int_bin"]):
                    if g in cms:
                        cms[g] += cm
                s = ts.summarize(cm)
                per_img[k][name + "_miou"] = s["miou_parts"]
                per_img[k][name + "_handle"] = None if math.isnan(s["iou"]["handle"]) else s["iou"]["handle"]
                per_img[k][name + "_taza"] = s["iou_mug"]
        res = {}
        for g in groups:
            if cnt[g] == 0:
                continue
            s = ts.summarize(cms[g])
            res[g] = (s["miou_parts"], s["iou"]["handle"], s["iou_mug"])
        hrow = cms["all"][HANDLE]
        hcol = cms["all"][:, HANDLE]
        res["_to"] = [float(v) / float(hrow.sum()) if hrow.sum() > 0 else float("nan") for v in hrow]
        res["_prec"] = float(cms["all"][HANDLE, HANDLE] / hcol.sum()) if hcol.sum() > 0 else float("nan")
        table[name] = res
        print("model evaluated: %s (%s)" % (name, mname))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    names = list(table)
    w = max(10, max(len(n) for n in names))
    for title, idx in (("parts mIoU", 0), ("handle IoU", 1), ("mug IoU", 2)):
        print("\n=== %s (%d px band) ===" % (title, a.band))
        print("%-16s %4s  " % ("grupo", "n") + "  ".join("%*s" % (w, n) for n in names))
        for g in groups:
            if cnt[g] == 0:
                continue
            print("%-16s %4d  " % (g, cnt[g]) + "  ".join("%*s" % (w, fmt(table[n][g][idx])) for n in names))
    print("\n=== annotated HANDLE pixels: which class the model assigns them to (all photos) ===")
    print("%-*s  " % (w, "model") + "  ".join("%9s" % c for c in ts.PARTS) + "   handle precision")
    for n in names:
        print("%-*s  " % (w, n) + "  ".join("%8.1f%%" % (100 * v) for v in table[n]["_to"]) +
              "   %.3f" % table[n]["_prec"])

    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(per_img[0].keys()))
        wr.writeheader()
        wr.writerows(per_img)
    print("\nCSV per photo: %s" % os.path.abspath(a.csv))


if __name__ == "__main__":
    main()
