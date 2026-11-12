# =========================================================================
# real_masks  v1.1   (external Python: Isaac's python.bat; numpy + Pillow)
# Converts X-AnyLabeling annotations (one .json per photo) into the REAL test
# set, with the same format and the same part ids as the synthetic dataset.
#
# Reads:  <annotate>/IMG_xxxx.jpeg + IMG_xxxx.json   (X-AnyLabeling polygons)
#         <annotate>/real_mugs*.csv  (OPTIONAL: photo, mug_id, shape, material)
# Writes (--out; default from the annotation folder name:
#         annotate -> real_test, annotate_final -> real_test_final,
#         annotate_XXX -> real_test_XXX, any other name -> <name>_out):
#   rgb/IMG_xxxx.png        1280x720 (or --width), sRGB
#   part/IMG_xxxx.png       8-bit: 0 background, 1 body, 2 handle, 3 rim, 4 interior,
#                           5 base, 255 ignore (not evaluated)
#   preview/IMG_xxxx.jpg    photo + colour mask + contours (for review)
#   meta/IMG_xxxx.json      K (from EXIF), sizes, pixels per part, warnings
#   index.csv
# Safety: the output folder stores source.txt with its annotation folder; if it
# belongs to ANOTHER annotation folder the script stops (unless --force). Outputs
# of photos that no longer have a .json are removed.
#
# Rules (the ones used when annotating):
#   - painting priority: body < interior < base < rim < handle < ignore
#   - mug silhouette = body + interior; rim and base are CLIPPED against it
#     (they can be drawn generously outwards; the SAM silhouette wins)
#   - the handle is not clipped (it lies outside the body)
#   - polygons rasterised directly at the output resolution, without
#     anti-aliasing: exact values at edges
#   - portrait photos are skipped (the dataset is 16:9 landscape)
# K: 35 mm-equivalent focal length from EXIF (43.27 mm diagonal) over the full
#    sensor diagonal (ExifImageWidth x ExifImageHeight); the iPhone's 16:9 is a
#    centred crop, so cx, cy = centre. OpenCV convention (pixel 0 at 0).
#
# Usage:
#   python.bat real_masks.py --annotate <SYNTH_ROOT>/real_photos/annotate
# =========================================================================
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

try:
    from PIL import Image, ImageDraw, ExifTags, ImageOps
except ImportError:
    sys.exit("Pillow is missing:  python.bat -m pip install pillow")

VERSION = "v1.1"
PART_IDS = {"body": 1, "handle": 2, "rim": 3, "interior": 4, "base": 5, "ignore": 255}
ORDER = ["body", "interior", "base", "rim", "handle", "ignore"]      # bottom to top
PAL = np.zeros((256, 3), np.uint8)
PAL[1], PAL[2], PAL[3], PAL[4], PAL[5], PAL[255] = (150, 150, 150), (230, 30, 30), (30, 200, 30), \
    (230, 200, 30), (40, 80, 230), (255, 0, 255)
TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def exif_K(img, out_w, out_h):
    ex = img.getexif()
    sub = ex.get_ifd(0x8769)
    f35 = sub.get(TAGS["FocalLengthIn35mmFilm"]) or ex.get(TAGS["FocalLengthIn35mmFilm"])
    fw = sub.get(TAGS["ExifImageWidth"]) or img.size[0]
    fh = sub.get(TAGS["ExifImageHeight"]) or img.size[1]
    if not f35:
        return None, None
    full_w, full_h = max(fw, fh), min(fw, fh)                  # full sensor, landscape
    f_full = float(f35) / 43.2666 * math.hypot(full_w, full_h)  # px at full resolution
    fx = f_full * out_w / float(max(img.size))                  # same horizontal width (vertical crop)
    K = [[fx, 0.0, out_w / 2.0 - 0.5], [0.0, fx, out_h / 2.0 - 0.5], [0.0, 0.0, 1.0]]
    hfov = math.degrees(2 * math.atan(out_w / (2 * fx)))
    return K, hfov


def raster(shapes, label, sx, sy, W, H):
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    n = 0
    for s in shapes:
        if s.get("label") != label:
            continue
        st = s.get("shape_type", "polygon")
        pts = [(x * sx, y * sy) for x, y in s.get("points", [])]
        if st == "rectangle" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1] if len(pts) == 2 else pts[2]
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=1)
        elif len(pts) >= 3:
            d.polygon(pts, fill=1)
        else:
            continue
        n += 1
    return np.array(m, dtype=bool), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--mugs_csv", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ann = args.annotate
    ann = os.path.normpath(ann)
    base = os.path.basename(ann)
    if base == "annotate":
        default_out = "real_test"
    elif base.startswith("annotate_"):
        default_out = "real_test_" + base[len("annotate_"):]
    else:
        default_out = base + "_out"
    out = os.path.normpath(args.out or os.path.join(os.path.dirname(ann), default_out))
    if out == ann:
        sys.exit("--out cannot be the annotation folder itself")
    src = os.path.join(out, "source.txt")
    if os.path.isfile(src) and not args.force:
        with open(src, encoding="utf-8") as fh:
            prev = fh.read().strip()
        if os.path.normcase(prev) != os.path.normcase(ann):
            sys.exit("STOP: %s belongs to another annotation folder (%s).\n"
                     "Use another --out, or --force if you really want to replace it." % (out, prev))
    print("annotations: %s\noutput:      %s" % (ann, out))
    for s in ("rgb", "part", "preview", "meta"):
        os.makedirs(os.path.join(out, s), exist_ok=True)

    mugs = {}
    mp = args.mugs_csv
    if not mp:
        cands = [os.path.join(ann, n) for n in ("real_mugs_final.csv", "real_mugs.csv")]
        cands += sorted(glob.glob(os.path.join(ann, "real_mugs*.csv")))
        mp = next((c for c in cands if os.path.isfile(c)), "")
    print("mugs:        %s" % (mp or "(no mug CSV)"))
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                mugs[os.path.splitext(r.get("foto", r.get("photo", "")))[0]] = r

    rows = []
    for jp in sorted(glob.glob(os.path.join(ann, "*.json"))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        if stem.startswith("real_mugs") or stem == "mask_grayscale_map":
            continue
        with open(jp, encoding="utf-8") as fh:
            a = json.load(fh)
        if "shapes" not in a:
            continue
        ip = os.path.join(ann, a.get("imagePath") or "")
        if not os.path.isfile(ip):
            c = [p for p in glob.glob(os.path.join(ann, stem + ".*")) if not p.endswith(".json")]
            ip = c[0] if c else ""
        if not ip:
            print("[warning] %s: photo not found" % stem)
            continue
        img = Image.open(ip)
        warn = []
        iw, ih = img.size
        aw, ah = int(a.get("imageWidth") or iw), int(a.get("imageHeight") or ih)
        if (aw, ah) != (iw, ih):
            warn.append("json size %dx%d != photo %dx%d" % (aw, ah, iw, ih))
        if ih > iw:
            print("[skip] %s: portrait photo" % stem)
            continue
        W = args.width
        H = int(round(W * ih / float(iw)))
        sx, sy = W / float(aw), H / float(ah)
        K, hfov = exif_K(img, W, H)
        if K is None:
            warn.append("no focal length in EXIF")

        labels = {s.get("label") for s in a["shapes"]}
        unknown = sorted(l for l in labels if l not in PART_IDS)
        if unknown:
            warn.append("unknown labels: %s" % ", ".join(unknown))

        masks, counts = {}, {}
        for lab in ORDER:
            masks[lab], counts[lab] = raster(a["shapes"], lab, sx, sy, W, H)
        sil = masks["body"] | masks["interior"]
        if not masks["body"].any():
            warn.append("no body: rim/base cannot be clipped")
            sil = np.ones((H, W), bool)
        part = np.zeros((H, W), np.uint8)
        clipped = {}
        for lab in ORDER:
            m = masks[lab]
            if lab in ("rim", "base", "interior"):
                clipped[lab] = int((m & ~sil).sum())
                m = m & sil
            part[m] = PART_IDS[lab]

        rgb = ImageOps.exif_transpose(img).convert("RGB").resize((W, H), Image.LANCZOS)
        rgb.save(os.path.join(out, "rgb", stem + ".png"))
        Image.fromarray(part).save(os.path.join(out, "part", stem + ".png"))

        # review image
        r = np.asarray(rgb, dtype=np.float32)
        sel = part > 0
        r[sel] = 0.55 * r[sel] + 0.45 * PAL[part[sel]]
        e = np.zeros((H, W), bool)
        e[1:, :] |= part[1:, :] != part[:-1, :]
        e[:, 1:] |= part[:, 1:] != part[:, :-1]
        r[e] = (255, 0, 255)
        Image.fromarray(r.astype(np.uint8)).save(os.path.join(out, "preview", stem + ".jpg"), quality=88)

        px = {lab: int((part == PART_IDS[lab]).sum()) for lab in PART_IDS}
        mi = mugs.get(stem, {})
        meta = {
            "real_masks_version": VERSION, "photo": os.path.basename(ip), "width": W, "height": H,
            "K_opencv": K, "hfov_deg": hfov, "part_ids": PART_IDS, "part_px": px,
            "shapes_per_label": counts, "clipped_px_outside_silhouette": clipped,
            "mug_id": mi.get("mug_id"), "shape": mi.get("shape"), "material": mi.get("material"),
            "warnings": warn,
        }
        with open(os.path.join(out, "meta", stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
        rows.append({"photo": stem, "mug_id": mi.get("mug_id", ""), "shape": mi.get("shape", ""),
                     "material": mi.get("material", ""), **{k + "_px": v for k, v in px.items()},
                     "rim_clipped_px": clipped.get("rim", 0), "base_clipped_px": clipped.get("base", 0),
                     "hfov_deg": round(hfov, 2) if hfov else "", "warnings": " | ".join(warn)})
        print("%s  body %6d  handle %5d  rim %5d  interior %5d  base %5d  ignore %5d  %s" %
              (stem, px["body"], px["handle"], px["rim"], px["interior"], px["base"], px["ignore"],
               ("WARNING: " + "; ".join(warn)) if warn else ""))

    if not rows:
        sys.exit("No .json annotations found in %s" % ann)
    # remove leftovers of photos no longer annotated in this folder
    keep = {r["photo"] for r in rows}
    removed = 0
    for sub, ext in (("rgb", ".png"), ("part", ".png"), ("preview", ".jpg"), ("meta", ".json")):
        for f in glob.glob(os.path.join(out, sub, "*" + ext)):
            if os.path.splitext(os.path.basename(f))[0] not in keep:
                os.remove(f)
                removed += 1
    if removed:
        print("removed %d files of photos without annotation" % removed)
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(ann)
    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n%d photos -> %s   (check preview/)" % (len(rows), out))


if __name__ == "__main__":
    main()
