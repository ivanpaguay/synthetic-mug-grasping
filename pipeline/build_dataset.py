# =========================================================================
# build_dataset  v1.2   (external Python: Isaac's python.bat; numpy + OpenEXR + Pillow)
# Converts the rendered EXRs + per-image metadata (image_metadata.py) into the
# final dataset.
#
# Reads:  $HIP/renders/scene_XXXXXX/vV_gG.exr and vV_gG.json (from image_metadata.py)
#         $HIP/scenes/scene_XXXXXX/scene.json
# Writes (--out, default $HIP/dataset_pilot1):
#   dataset_info.json        codes, conventions, scales, splits, known characteristics
#   index.csv                one row per image (split, flags, paths)
#   scenes/scene_XXXXXX/
#     scene.json
#     rgb/vV_gG.png          8-bit sRGB (linear beauty converted, clipped to [0,1])
#     depth/vV_gG.png        16-bit camera Z in units of 0.1 mm (0 = background)
#     instance/vV_gG.png     16-bit instance_id (65535 = background; 0 = table)
#     class/vV_gG.png        8-bit class_id (0 = background)
#     part/vV_gG.png         8-bit part_id (mugs only; 0 = not applicable)
#     normal/vV_gG.npz       float16 (H, W, 3), normal in OpenCV camera frame (x right, y down, z forward)
#     meta/vV_gG.json        per-image metadata + split + flags
#     exr/vV_gG.exr          (only with --keep_exr 1)
#
# Split by TARGET MUG (not by image): a mug is never in train and test at once.
#   --val_mugs 103,104  --test_mugs 107,108   (the rest, train)
# Flag: target_heavily_occluded = 1 if, in _g1, the mug keeps < --occ_thresh (0.25)
#   of its pixels compared with its _g0 pair (kept, for filtering).
# sem_ids stored as separate channels (sem_ids.r/.g/.b) are read in r,g,b order.
#
# Usage (from the Isaac folder):
#   python.bat build_dataset.py --hip <SYNTH_ROOT>/HIPS --out <SYNTH_ROOT>/HIPS/dataset_E
#       --val_mugs 103,104,105,106,124 --test_mugs 107,108,109,112,136 --scene_min 10000 --scene_max 13914
# Options: --scenes 0,100, --keep_exr 1, --normals 0
# =========================================================================
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

try:
    import OpenEXR
except ImportError:
    sys.exit("OpenEXR is missing:  python.bat -m pip install OpenEXR")
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is missing:  python.bat -m pip install pillow")

VERSION = "v1.2"
DEPTH_UNIT_M = 1e-4          # 0.1 mm per unit -> 6.55 m maximum
INSTANCE_BG = 65535


def read_exr(path):
    f = OpenEXR.File(path)
    ch = {}
    for p in f.parts:
        for name, c in p.channels.items():
            ch[name] = c.pixels
    rgb = None
    for k in ("RGBA", "RGB"):
        if k in ch:
            rgb = ch[k][..., :3]
    if rgb is None:
        rgb = np.stack([ch[k] for k in ("R", "G", "B")], -1)
    if "sem_ids" in ch and ch["sem_ids"].ndim == 3:
        S = ch["sem_ids"]
    else:
        comps = [k for k in ch if k.startswith("sem_ids.")]
        comps.sort(key=lambda k: "rgbxyz".find(k.split(".")[-1].lower()))   # r,g,b (not alphabetical)
        S = np.stack([ch[k] for k in comps[:3]], -1)
    S = np.rint(S).astype(np.int32)
    dk = [k for k in ch if k == "depth" or k.startswith("depth.")][0]
    D = ch[dk].astype(np.float64)
    if D.ndim == 3:
        D = D[..., 0]
    nk = sorted(k for k in ch if k.startswith("N_world."))
    N = np.stack([ch[k] for k in nk], -1).astype(np.float32) if len(nk) == 3 else None
    return rgb.astype(np.float32), S, D, N


def linear_to_srgb8(c):
    c = np.clip(np.nan_to_num(c), 0.0, 1.0)
    s = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return (np.clip(s, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png(arr, path):
    if arr.dtype == np.uint16:
        Image.fromarray(arr, mode="I;16").save(path, optimize=False)
    else:
        Image.fromarray(arr).save(path, optimize=False)


def parse_ids(s):
    return {int(x) for x in s.split(",") if x.strip() != ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--keep_exr", type=int, default=0)
    ap.add_argument("--normals", type=int, default=1)
    ap.add_argument("--val_mugs", default="8")
    ap.add_argument("--test_mugs", default="9")
    ap.add_argument("--occ_thresh", type=float, default=0.25)
    ap.add_argument("--scene_min", type=int, default=-1)
    ap.add_argument("--scene_max", type=int, default=-1)
    args = ap.parse_args()

    out = args.out or os.path.join(args.hip, "dataset_pilot1")
    val_m, test_m = parse_ids(args.val_mugs), parse_ids(args.test_mugs)
    rdir = os.path.join(args.hip, "renders")
    sdirs = sorted(glob.glob(os.path.join(rdir, "scene_*")))
    if args.scenes:
        want = parse_ids(args.scenes)
        sdirs = [d for d in sdirs if int(os.path.basename(d)[6:]) in want]
    if args.scene_min >= 0:
        sdirs = [d for d in sdirs if int(os.path.basename(d)[6:]) >= args.scene_min]
    if args.scene_max >= 0:
        sdirs = [d for d in sdirs if int(os.path.basename(d)[6:]) <= args.scene_max]
    print("scenes to process: %d" % len(sdirs))

    rows = []
    class_ids = part_ids = None
    size = None
    for sd in sdirs:
        sid = int(os.path.basename(sd)[6:])
        sj = os.path.join(args.hip, "scenes", "scene_%06d" % sid, "scene.json")
        if not os.path.isfile(sj):
            print("[warning] missing %s" % sj)
            continue
        with open(sj, encoding="utf-8") as fh:
            scene = json.load(fh)
        class_ids = class_ids or scene.get("class_ids")
        part_ids = part_ids or scene.get("part_ids")
        mug = int(scene.get("target_mug_id", -1))
        split = "test" if mug in test_m else "val" if mug in val_m else "train"
        od = os.path.join(out, "scenes", "scene_%06d" % sid)
        subs = ["rgb", "depth", "instance", "class", "part", "meta"]
        if args.normals:
            subs.append("normal")
        if args.keep_exr:
            subs.append("exr")
        for s in subs:
            os.makedirs(os.path.join(od, s), exist_ok=True)
        shutil.copyfile(sj, os.path.join(od, "scene.json"))

        for ep in sorted(glob.glob(os.path.join(sd, "v*_g*.exr"))):
            stem = os.path.basename(ep)[:-4]
            mj = ep[:-4] + ".json"
            if not os.path.isfile(mj):
                print("[warning] missing %s (run image_metadata.py first)" % mj)
                continue
            with open(mj, encoding="utf-8") as fh:
                meta = json.load(fh)
            rgb, S, D, N = read_exr(ep)
            H, W = D.shape
            size = [W, H]

            save_png(linear_to_srgb8(rgb), os.path.join(od, "rgb", stem + ".png"))
            dq = np.clip(np.rint(D / DEPTH_UNIT_M), 0, 65535).astype(np.uint16)
            save_png(dq, os.path.join(od, "depth", stem + ".png"))
            inst = np.where(S[..., 1] > 0, S[..., 0], INSTANCE_BG).astype(np.uint16)
            save_png(inst, os.path.join(od, "instance", stem + ".png"))
            save_png(S[..., 1].astype(np.uint8), os.path.join(od, "class", stem + ".png"))
            save_png(S[..., 2].astype(np.uint8), os.path.join(od, "part", stem + ".png"))
            if args.normals and N is not None:
                R = np.array(meta["camera"]["world_to_cam_opencv"], dtype=np.float32)[:3, :3]
                nc = N @ R.T
                nc[S[..., 1] == 0] = 0.0
                np.savez_compressed(os.path.join(od, "normal", stem + ".npz"), normal=nc.astype(np.float16))
            if args.keep_exr:
                shutil.copyfile(ep, os.path.join(od, "exr", stem + ".exr"))

            tgt = [o for o in meta["objects"] if o["class_id"] == 2]
            tpx = tgt[0]["pixels"] if tgt else 0
            vis = tgt[0].get("visible_frac_vs_no_gripper") if tgt else (0.0 if meta["with_gripper"] else None)
            heavy = int(meta["with_gripper"] == 1 and (vis is None or vis < args.occ_thresh))
            meta["dataset"] = {
                "build_dataset_version": VERSION, "split": split,
                "target_heavily_occluded": heavy, "target_visible_frac": vis,
                "files": {k: "%s/%s.%s" % (k, stem, "npz" if k == "normal" else "exr" if k == "exr" else "png")
                          for k in subs if k != "meta"},
            }
            with open(os.path.join(od, "meta", stem + ".json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1, ensure_ascii=False)

            rows.append({
                "split": split, "scene_id": sid, "view_id": meta["view_id"],
                "with_gripper": meta["with_gripper"], "target_mug_id": mug,
                "grasp_sample_id": meta.get("grasp_sample_id"),
                "grasp_final_tasks": meta.get("grasp_final_tasks"),
                "scene_style": scene.get("scene_style"),
                "target_px": tpx, "target_visible_frac": "" if vis is None else vis,
                "target_heavily_occluded": heavy,
                "handle_px": meta.get("target_parts_px", {}).get("handle", 0),
                "n_objects": len(meta["objects"]),
                "rgb": "scenes/scene_%06d/rgb/%s.png" % (sid, stem),
                "meta": "scenes/scene_%06d/meta/%s.json" % (sid, stem),
            })
        print("scene %06d (%s)" % (sid, split))

    if not rows:
        sys.exit("No image was processed")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        c = counts.setdefault(r["split"], {"images": 0, "scenes": set(), "mugs": set(), "heavily_occluded": 0})
        c["images"] += 1
        c["scenes"].add(r["scene_id"])
        c["mugs"].add(r["target_mug_id"])
        c["heavily_occluded"] += r["target_heavily_occluded"]
    counts = {k: {"images": v["images"], "scenes": len(v["scenes"]), "mugs": sorted(v["mugs"]),
                  "heavily_occluded": v["heavily_occluded"]} for k, v in counts.items()}
    info = {
        "name": "mug_affordance_grasp_synthetic",
        "build_dataset_version": VERSION,
        "image_size": size,
        "class_ids": class_ids, "part_ids": part_ids,
        "encodings": {
            "rgb": "PNG 8-bit sRGB, from the linear Karma beauty clipped to [0,1]",
            "depth": "PNG 16-bit camera Z; metres = value * %g; 0 = background" % DEPTH_UNIT_M,
            "instance": "PNG 16-bit; instance_id (0 table, 1 target mug, 2.. distractors, 100 gripper); %d = background" % INSTANCE_BG,
            "class": "PNG 8-bit; class_ids; 0 = background",
            "part": "PNG 8-bit; part_ids (mugs only); 0 = not applicable",
            "normal": "NPZ float16 (H,W,3), key 'normal'; OpenCV camera frame; 0 on the background",
        },
        "camera_conventions": "K and world_to_cam_opencv in meta/*.json: OpenCV (x right, y down, z forward, "
                              "centre of pixel (0,0) at (0,0)); Houdini world Y up, metres",
        "pairs": "each view has _g1 (with gripper) and _g0 (without gripper), same camera and same scene",
        "split_rule": "by target mug; val=%s test=%s" % (sorted(val_m), sorted(test_m)),
        "occlusion_flag": "target_heavily_occluded: _g1 keeps < %.2f of the mug pixels of _g0" % args.occ_thresh,
        "known_characteristics": [
            "masks and depth use the nearest sample (no averaging): exact ids at edges",
            "depth biased ~0.4-1.3 mm towards the camera on slanted surfaces (nearest sample within the pixel)",
            "between _g0 and _g1 ~0.01% of isolated silhouette pixels may change (sampling noise)",
            "the mug handle is a grasp negative in this version (see the pilot report)",
            "small mugs (espresso, tea) have few valid grasps: the gripper hits the table; a grasp can repeat across scenes",
        ],
        "counts": counts,
    }
    with open(os.path.join(out, "dataset_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, ensure_ascii=False)

    print("\n=== dataset ===")
    for k in ("train", "val", "test"):
        if k in counts:
            c = counts[k]
            print("%-5s %4d images  %3d scenes  mugs %s  heavily occluded %d" %
                  (k, c["images"], c["scenes"], c["mugs"], c["heavily_occluded"]))
    tot = 0
    for root, _, files in os.walk(out):
        tot += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print("size: %.1f MB  ->  %s" % (tot / 1e6, out))


if __name__ == "__main__":
    main()
