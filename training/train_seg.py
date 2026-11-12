# =========================================================================
# train_seg  v1.5   (training venv: PyTorch + transformers)
# Trains a mug PART segmenter (SegFormer) and evaluates it on the synthetic
# val/test splits and on REAL photos.
# Classes: 0 background, 1 body, 2 handle, 3 rim, 4 interior, 5 base  (255 = ignore)
#
# Inputs:
#   --data        synthetic dataset (index.csv + scenes/.../rgb, part); "" = none
#   --real        real set(s) for evaluation, comma-separated (index.csv + rgb/ + part/)
#   --train_real  real set added to TRAINING (real-only and fine-tune runs);
#                 a folder in --train_real can NEVER also be in --real
# Output: --out (runs/<name>): log.txt, metrics.json, best.pt, preview_<real set>/
#
# Model selection: by mIoU on the SYNTHETIC VALIDATION split (--select val), or
# the LAST iteration (--select last, for real-only and fine-tuned models). Real
# photos never choose anything.
#
# Metrics: IoU per part, mIoU of the 5 parts, "exterior" (body+rim+base merged),
# "mug" (any part vs background). Two variants: strict, and
# with a --band px tolerance band (at 1280 width) along every class boundary.
# Real-set groups: main (no wide angle), in_family (ceramic, not novelty, no wide
# angle), shape_*, material_*, wide_angle.
#
# Other options:
#   --init CHECKPOINT, --eval_only 1          evaluate an existing model (no --data needed)
#   --train_n N                               N random training images (fixed seed 123),
#                                             to compare subsets of equal size
#   --real_rep N                              repeat each real training photo N times
#   --handle_focus P, --class_w "w0,...,w5"   crop on handle pixels / class weights
#   --model nvidia/mit-b0 | nvidia/mit-b2, --iters 8000, --batch 8, --crop 512,
#   --gripper both|with|without, --style both|realistic|random, --drop_heavy 1,
#   --eval_every 500, --workers 4
#
# Usage:
#   python train_seg.py --data <SYNTH_ROOT>/HIPS/dataset_E
#       --real <SYNTH_ROOT>/real_photos/real_test --out runs/E_b0
#   python train_seg.py --data <SYNTH_ROOT>/HIPS/dataset_E --real <...>/real_test
#       --out runs/E_b2 --model nvidia/mit-b2 --iters 32000 --batch 4
#   python train_seg.py --init runs/E_b2/best.pt --train_real <...>/real_test
#       --out runs/FT_b2 --model nvidia/mit-b2 --iters 3000 --batch 4 --lr 2e-5 --select last
# (B2 at batch 8 does not fit in 8 GB of VRAM: use batch 4.)
# =========================================================================
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

VERSION = "v1.5"
PARTS = ["background", "body", "handle", "rim", "interior", "base"]
NC = len(PARTS)
IGN = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- data
def read_csv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def synth_items(root, split, a):
    out = []
    for r in read_csv(os.path.join(root, "index.csv")):
        if r["split"] != split:
            continue
        g = int(r["with_gripper"])
        if a.gripper == "with" and g != 1 or a.gripper == "without" and g != 0:
            continue
        st = r.get("scene_style", "")
        if a.style == "realistic" and st != "0" or a.style == "random" and st != "1":
            continue
        if a.drop_heavy and r.get("target_heavily_occluded") == "1":
            continue
        rgb = os.path.join(root, r["rgb"])
        out.append({"rgb": rgb, "part": rgb.replace(os.sep + "rgb" + os.sep, os.sep + "part" + os.sep)
                                          .replace("/rgb/", "/part/"),
                    "name": "%s_v%s_g%s" % (r["scene_id"], r["view_id"], r["with_gripper"]),
                    "groups": ["gripper_%s" % r["with_gripper"]]})
    return out


def real_items(root):
    rows = {r["photo"]: r for r in read_csv(os.path.join(root, "index.csv"))}
    out = []
    for name, r in sorted(rows.items()):
        shape = (r.get("shape") or "unknown").strip() or "unknown"
        mat = (r.get("material") or "unknown").strip() or "unknown"
        try:
            hfov = float(r.get("hfov_deg") or 0)
        except ValueError:
            hfov = 0.0
        groups = ["shape_" + shape, "material_" + mat]
        if hfov > 90:
            groups.append("wide_angle")
        else:
            groups.append("main")
            if mat == "ceramic" and shape != "novelty":
                groups.append("in_family")
        out.append({"rgb": os.path.join(root, "rgb", name + ".png"),
                    "part": os.path.join(root, "part", name + ".png"),
                    "name": name, "groups": groups, "mug_id": r.get("mug_id", "")})
    return out


def to_tensor(img):
    a = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


class TrainSet(Dataset):
    def __init__(self, items, crop, focus, smin, smax, handle_focus=0.0):
        self.items, self.crop, self.focus, self.smin, self.smax = items, crop, focus, smin, smax
        self.handle_focus = handle_focus
        from torchvision.transforms import ColorJitter
        self.jit = ColorJitter(0.4, 0.4, 0.4, 0.05)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["rgb"]).convert("RGB")
        m = Image.open(it["part"])
        s = random.uniform(self.smin, self.smax)
        W, H = img.size
        nw, nh = max(1, int(W * s)), max(1, int(H * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        m = m.resize((nw, nh), Image.NEAREST)
        c = self.crop
        if nw < c or nh < c:                                 # pad: black image, ignored mask
            pi = Image.new("RGB", (max(nw, c), max(nh, c)))
            pm = Image.new("L", (max(nw, c), max(nh, c)), IGN)
            pi.paste(img, (0, 0))
            pm.paste(m, (0, 0))
            img, m, nw, nh = pi, pm, max(nw, c), max(nh, c)
        a = np.asarray(m)
        ys, xs = np.nonzero((a > 0) & (a != IGN))
        if self.handle_focus > 0 and random.random() < self.handle_focus:
            hy, hx = np.nonzero(a == 2)
            if len(hy):
                ys, xs = hy, hx
        if len(ys) and random.random() < self.focus:        # crop centred near the mug
            k = random.randrange(len(ys))
            cx = xs[k] + random.randint(-c // 3, c // 3)
            cy = ys[k] + random.randint(-c // 3, c // 3)
            x0 = int(np.clip(cx - c // 2, 0, nw - c))
            y0 = int(np.clip(cy - c // 2, 0, nh - c))
        else:
            x0, y0 = random.randint(0, nw - c), random.randint(0, nh - c)
        img = img.crop((x0, y0, x0 + c, y0 + c))
        m = m.crop((x0, y0, x0 + c, y0 + c))
        if random.random() < 0.5:
            img, m = img.transpose(Image.FLIP_LEFT_RIGHT), m.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.jit(img)
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        x = to_tensor(img)
        if random.random() < 0.3:
            x = x + torch.randn_like(x) * random.uniform(0.02, 0.08) / 0.226
        return x, torch.from_numpy(np.asarray(m, np.int64).copy())


class EvalSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["rgb"]).convert("RGB")
        m = Image.open(it["part"])
        if m.size != img.size:
            m = m.resize(img.size, Image.NEAREST)
        return to_tensor(img), torch.from_numpy(np.asarray(m, np.int64).copy()), i


# ---------------------------------------------------------------- model
def build_model(name):
    from transformers import SegformerForSemanticSegmentation
    id2label = {i: n for i, n in enumerate(PARTS)}
    return SegformerForSemanticSegmentation.from_pretrained(
        name, num_labels=NC, id2label=id2label, label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True)


def logits_full(model, x):
    out = model(pixel_values=x).logits
    return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------- metrics
def void_band(y, px):
    """Pixels closer than px to a boundary between different classes (background included)."""
    if px <= 0:
        return torch.zeros_like(y, dtype=torch.bool)
    e = torch.zeros_like(y, dtype=torch.bool)
    e[:, 1:, :] |= y[:, 1:, :] != y[:, :-1, :]
    e[:, :, 1:] |= y[:, :, 1:] != y[:, :, :-1]
    k = 2 * px + 1
    return F.max_pool2d(e.float().unsqueeze(1), k, 1, px).squeeze(1) > 0


def confusion(pred, y, valid):
    p, t = pred[valid], y[valid]
    return torch.bincount(t * NC + p, minlength=NC * NC).reshape(NC, NC).cpu().double()


def iou_from(cm):
    tp = cm.diag()
    den = cm.sum(0) + cm.sum(1) - tp
    return [float(tp[i] / den[i]) if den[i] > 0 else float("nan") for i in range(NC)]


def merged_iou(cm, groups):
    """IoU of the union of classes 'groups' against the rest."""
    g = list(groups)
    o = [i for i in range(NC) if i not in g]
    tp = cm[g][:, g].sum()
    fn = cm[g][:, o].sum()
    fp = cm[o][:, g].sum()
    d = tp + fn + fp
    return float(tp / d) if d > 0 else float("nan")


def summarize(cm):
    iou = iou_from(cm)
    parts = [v for v in iou[1:] if not math.isnan(v)]
    return {"iou": {PARTS[i]: round(iou[i], 4) for i in range(NC)},
            "miou_parts": round(float(np.mean(parts)), 4) if parts else None,
            "iou_exterior": round(merged_iou(cm, (1, 3, 5)), 4),
            "iou_mug": round(merged_iou(cm, (1, 2, 3, 4, 5)), 4)}


@torch.no_grad()
def evaluate(model, items, dev, band_px, amp, preview_dir=None, n_prev=0):
    model.eval()
    dl = DataLoader(EvalSet(items), batch_size=1, shuffle=False, num_workers=2)
    cms = {}
    for x, y, idx in dl:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            lg = logits_full(model, x)
        pred = lg.argmax(1)
        valid = y != IGN
        band = void_band(y.masked_fill(~valid, 0), max(1, round(band_px * y.shape[-1] / 1280.0)))
        grps = ["all"] + items[int(idx)]["groups"]
        for key, v in (("strict", valid), ("band", valid & ~band)):
            for g in grps:
                cms.setdefault((key, g), torch.zeros(NC, NC, dtype=torch.float64))
                cms[(key, g)] += confusion(pred, y, v)
        if preview_dir and int(idx) < n_prev:
            save_preview(x[0], y[0], pred[0], os.path.join(preview_dir, items[int(idx)]["name"] + ".jpg"))
    model.train()
    res = {}
    for (key, g), cm in cms.items():
        res.setdefault(g, {})[key] = summarize(cm)
    return res


PAL = np.zeros((256, 3), np.uint8)
PAL[1], PAL[2], PAL[3], PAL[4], PAL[5], PAL[255] = (150, 150, 150), (230, 30, 30), (30, 200, 30), \
    (230, 200, 30), (40, 80, 230), (255, 0, 255)


def save_preview(x, y, p, path):
    img = (x.cpu().numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1) * 255
    def over(lbl):
        l = lbl.cpu().numpy().astype(np.uint8)
        o = img.copy()
        s = l > 0
        o[s] = 0.5 * o[s] + 0.5 * PAL[l[s]]
        return o
    both = np.concatenate([over(y), over(p)], 1).astype(np.uint8)    # left: annotation, right: model
    Image.fromarray(both).resize((both.shape[1] // 2, both.shape[0] // 2)).save(path, quality=85)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--real", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nvidia/mit-b0")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--focus", type=float, default=0.7)
    ap.add_argument("--scale_min", type=float, default=0.75)
    ap.add_argument("--scale_max", type=float, default=1.5)
    ap.add_argument("--gripper", default="both", choices=["both", "with", "without"])
    ap.add_argument("--style", default="both", choices=["both", "realistic", "random"])
    ap.add_argument("--drop_heavy", type=int, default=0)
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--init", default="")
    ap.add_argument("--handle_focus", type=float, default=0.0)
    ap.add_argument("--train_n", type=int, default=0)
    ap.add_argument("--train_real", default="")
    ap.add_argument("--real_rep", type=int, default=1)
    ap.add_argument("--select", default="val", choices=["val", "last"])
    ap.add_argument("--class_w", default="")
    ap.add_argument("--eval_only", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_preview", type=int, default=40)
    ap.add_argument("--max_items", type=int, default=0)       # only for quick tests
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "a", encoding="utf-8")

    def log(s):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if a.data:
        tr, va, te = (synth_items(a.data, s, a) for s in ("train", "val", "test"))
    else:
        tr, va, te = [], [], []
    if a.max_items:
        tr, va, te = tr[:a.max_items], va[:a.max_items], te[:a.max_items]
    if a.train_n and a.train_n < len(tr):
        tr = random.Random(123).sample(tr, a.train_n)
    n_synth = len(tr)
    eval_paths = [os.path.normcase(os.path.abspath(p.strip())) for p in a.real.split(",") if p.strip()]
    n_real_tr = 0
    if a.train_real:
        if os.path.normcase(os.path.abspath(a.train_real)) in eval_paths:
            raise SystemExit("--train_real cannot also be in --real (that would evaluate on training photos)")
        rt = real_items(a.train_real)
        n_real_tr = len(rt)
        tr = tr + rt * max(1, a.real_rep)
    if a.select == "last":
        va = []                                   # no selection by val: the last iteration is kept
    reals = {}
    for rp in [p.strip() for p in a.real.split(",") if p.strip()]:
        reals[os.path.basename(os.path.normpath(rp))] = real_items(rp)
    log("train_seg %s | model %s | train %d (synthetic %d + real %d x%d)  val %d  test %d  real %s | "
        "gripper=%s style=%s drop_heavy=%d select=%s" %
        (VERSION, a.model, len(tr), n_synth, n_real_tr, max(1, a.real_rep) if n_real_tr else 0, len(va), len(te),
         {k: len(v) for k, v in reals.items()}, a.gripper, a.style, a.drop_heavy, a.select))
    if not tr and not a.eval_only:
        raise SystemExit("no training images (is the --data path right?)")

    model = build_model(a.model).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
        log("initial weights: %s" % a.init)
    head = [p for n, p in model.named_parameters() if "decode_head" in n]
    body = [p for n, p in model.named_parameters() if "decode_head" not in n]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * 10}], weight_decay=0.01)
    warm = min(300, a.iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else max(0.0, 1 - (i - warm) / max(1, a.iters - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    cw = None
    if a.class_w:
        cw = torch.tensor([float(v) for v in a.class_w.split(",")], dtype=torch.float32, device=dev)
        assert len(cw) == NC, "--class_w needs %d values" % NC
    dl = None
    if not a.eval_only:                       # v1.5: --eval_only works without --data
        dl = DataLoader(TrainSet(tr, a.crop, a.focus, a.scale_min, a.scale_max, a.handle_focus),
                        batch_size=a.batch, shuffle=True,
                        num_workers=a.workers, drop_last=len(tr) >= a.batch, persistent_workers=a.workers > 0,
                        pin_memory=amp)
    best, best_it = -1.0, 0
    it, t0, run_loss = 0, time.time(), 0.0
    model.train()
    if a.eval_only:
        torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    while it < a.iters and not a.eval_only:
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                lg = logits_full(model, x)
                loss = F.cross_entropy(lg.float(), y, weight=cw, ignore_index=IGN)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            run_loss += float(loss.detach())
            if it % 50 == 0:
                el = time.time() - t0
                log("it %d/%d  loss %.4f  %.2f s/it  ~%.0f min left" %
                    (it, a.iters, run_loss / 50, el / it, el / it * (a.iters - it) / 60))
                run_loss = 0.0
            if va and (it % a.eval_every == 0 or it == a.iters):
                r = evaluate(model, va, dev, a.band, amp)["all"]["band"]
                log("  synthetic val: mIoU parts %.4f | exterior %.4f | mug %.4f | %s" %
                    (r["miou_parts"], r["iou_exterior"], r["iou_mug"],
                     " ".join("%s %.3f" % (k, v) for k, v in r["iou"].items() if k != "background")))
                if r["miou_parts"] is not None and r["miou_parts"] > best:
                    best, best_it = r["miou_parts"], it
                    torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
            if it >= a.iters:
                break
    if not va and not a.eval_only:
        torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    if va:
        log("best synthetic val: mIoU %.4f at it %d" % (best, best_it))
    else:
        best_it = it
        log("model saved: last iteration (%d)" % it)

    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev))
    metrics = {"version": VERSION, "args": vars(a), "best_val_miou": best, "best_iter": best_it,
               "n": {"train": len(tr), "val": len(va), "test": len(te),
                     "real": {k: len(v) for k, v in reals.items()}}}
    if te:
        metrics["synthetic_test"] = evaluate(model, te, dev, a.band, amp)
    for name, items in reals.items():
        pdir = os.path.join(a.out, "preview_" + name)
        os.makedirs(pdir, exist_ok=True)
        metrics["real_" + name] = evaluate(model, items, dev, a.band, amp, pdir, a.n_preview)
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, ensure_ascii=False)

    log("=== results (with a %d px band) ===" % a.band)
    def line(tag, r):
        log("%-34s mIoU %.3f | exterior %.3f | mug %.3f | %s" %
            (tag, r["miou_parts"] if r["miou_parts"] is not None else float("nan"),
             r["iou_exterior"], r["iou_mug"],
             " ".join("%s %.3f" % (k, v) for k, v in r["iou"].items() if k != "background")))
    if te:
        line("synthetic test", metrics["synthetic_test"]["all"]["band"])
    order = ["main", "in_family", "all", "wide_angle"]
    for name in reals:
        res = metrics["real_" + name]
        keys = [k for k in order if k in res] + sorted(k for k in res if k not in order)
        for g in keys:
            line("%s %s" % (name, g), res[g]["band"])
        if "main" in res:
            line("%s main (strict)" % name, res["main"]["strict"])
    log("metrics: %s" % os.path.join(a.out, "metrics.json"))


if __name__ == "__main__":
    main()
