# =========================================================================
# image_metadata  v1.1   (external Python: Isaac's python.bat, or any Python
#                         with numpy + OpenEXR)
# One metadata JSON per rendered image + a summary CSV, with automatic checks.
#
# Reads, per scene:
#   $HIP/scenes/scene_XXXXXX/scene.json        (objects, mug pose, grasp, views)
#   $HIP/renders/scene_XXXXXX/vV_gG.exr        (C, sem_ids, depth, N_world)
#   <grasp_dir>/sample_XXXXXXX.json            (contacts c_pos and gripper_pose, mug frame)
# Writes:
#   $HIP/renders/scene_XXXXXX/vV_gG.json       (metadata of that image)
#   $HIP/renders/<--summary>  (default metadata_summary.csv; one row per image)
#
# Per image:
#   camera: K (OpenCV convention: centre of pixel (0,0) at (0,0)), world_to_cam_cv
#           (x right, y down, z forward), cam_to_world_usd (x right, y up,
#           -z forward), W/H, fx/cx from the EXR header compared with camera_views
#   objects: per visible instance: class, pixels, 2D box [x0,y0,x1,y1] (inclusive),
#           truncated (touches the border); in _g1: pixels hidden by the gripper (vs _g0)
#   target_parts: pixels per part of the target mug
#   grasp_keypoints: contacts (with finger and zone) + wrist ("connect"), in world and
#           projected (u, v, z), inside the image, visible (|z - depth| < 1 cm),
#           mask class at that pixel
#   checks: reprojected table top (mm above y = 0), fraction of contacts that
#           land on the mug or the gripper (must be ~1 if the chain is consistent)
#
# Usage (from the Isaac folder):
#   python.bat -m pip install OpenEXR            (once)
#   python.bat image_metadata.py --hip <SYNTH_ROOT>/HIPS --grasp_dir <SYNTH_ROOT>/HIPS/grasp_data_E
#       --scene_min 10000 --scene_max 13914 --summary metadata_summary_E.csv
# Options: --scenes 0,100 (only those), --vis_tol 0.01
# =========================================================================
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

try:
    import OpenEXR
except ImportError:
    sys.exit("OpenEXR is missing:  python.bat -m pip install OpenEXR")

VERSION = "v1.1"


# ---------------------------------------------------------------- EXR
def load_exr(path):
    f = OpenEXR.File(path)
    ch = {}
    header = None
    for p in f.parts:
        if header is None or p.name() in ("C", "rgba", "beauty"):
            header = p.header
        for name, c in p.channels.items():
            ch[name] = c.pixels
    # sem_ids: one 3-component channel, or three separate channels
    if "sem_ids" in ch and ch["sem_ids"].ndim == 3:
        S = ch["sem_ids"]
    else:
        comps = [k for k in ch if k.startswith("sem_ids.")]
        comps.sort(key=lambda k: "RGBXYZrgbxyz".find(k.split(".")[-1]) % 6)
        S = np.stack([ch[k] for k in comps[:3]], -1)
    S = np.rint(S).astype(np.int32)
    dk = [k for k in ch if k == "depth" or k.startswith("depth.")]
    D = ch[dk[0]].astype(np.float64)
    if D.ndim == 3:
        D = D[..., 0]
    N = None
    nk = sorted(k for k in ch if k.startswith("N_world."))
    if len(nk) == 3:
        N = np.stack([ch[k] for k in nk], -1)          # .x .y .z in alphabetical order
    W2C = np.array(header["worldToCamera"], dtype=np.float64)
    W2N = np.array(header["worldToNDC"], dtype=np.float64)
    return S, D, N, W2C, W2N


# ---------------------------------------------------------------- camera
class Camera:
    """Karma EXR header: row matrices (p_row @ M), camera x right, y up, z forward."""

    def __init__(self, W2C, W2N, W, H):
        self.W, self.H = W, H
        self.W2C = W2C
        C2N = np.linalg.inv(W2C) @ W2N
        self.sx, self.sy = C2N[0, 0], C2N[1, 1]
        self.ox, self.oy = C2N[2, 0], C2N[2, 1]
        self.fx = 0.5 * W * self.sx
        self.fy = 0.5 * H * self.sy
        self.cx = 0.5 * W * (1.0 + self.ox) - 0.5        # OpenCV: centre of pixel 0 at 0
        self.cy = 0.5 * H * (1.0 - self.oy) - 0.5
        M = W2C.T                                        # columns: cam = M @ world
        F = np.diag([1.0, -1.0, 1.0, 1.0])
        self.world_to_cam_cv = F @ M
        G = np.diag([1.0, 1.0, -1.0, 1.0])
        self.cam_to_world_usd = np.linalg.inv(G @ M)

    def K(self):
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]

    def project(self, pw):
        pw = np.atleast_2d(np.asarray(pw, dtype=np.float64))
        ph = np.c_[pw, np.ones(len(pw))] @ self.world_to_cam_cv.T
        z = ph[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * ph[:, 0] / z + self.cx
            v = self.fy * ph[:, 1] / z + self.cy
        return u, v, z

    def unproject_depth(self, D):
        H, W = D.shape
        ys, xs = np.mgrid[0:H, 0:W]
        x = (xs - self.cx) / self.fx * D
        y = (ys - self.cy) / self.fy * D
        pc = np.stack([x, y, D, np.ones_like(D)], -1)
        return pc @ np.linalg.inv(self.world_to_cam_cv).T


# ---------------------------------------------------------------- utilities
def as_points(v):
    if v is None:
        return np.zeros((0, 3))
    a = np.asarray(v, dtype=np.float64)
    return a.reshape(-1, 3)


def mug_to_world(p, yaw_deg, T):
    # same rotation as Houdini's qrotate(quaternion(radians(yaw), {0,1,0}))
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return p @ R.T + np.asarray(T, dtype=np.float64)


def find_grasp_json(hip, grasp_dir, sample_id):
    name = "sample_%07d.json" % int(sample_id)
    cands = [grasp_dir] if grasp_dir else []
    cands += [os.path.join(hip, "grasp_data_family"),
              os.path.join(hip, "grasp_data_family_physics_pilot1c")]
    for d in cands:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    hits = glob.glob(os.path.join(hip, "grasp_data_family*", "**", name), recursive=True)
    return hits[0] if hits else None


def object_stats(S):
    """Per instance (only pixels with class > 0): class, pixels, box, truncation."""
    H, W = S.shape[:2]
    out = {}
    fg = S[..., 1] > 0
    inst = S[..., 0][fg]
    cls = S[..., 1][fg]
    ys, xs = np.nonzero(fg)
    for iid in np.unique(inst):
        m = inst == iid
        cvals, ccnt = np.unique(cls[m], return_counts=True)
        x0, x1 = int(xs[m].min()), int(xs[m].max())
        y0, y1 = int(ys[m].min()), int(ys[m].max())
        out[int(iid)] = {
            "class_id": int(cvals[np.argmax(ccnt)]),
            "class_ids_in_mask": len(cvals),
            "pixels": int(m.sum()),
            "bbox_xyxy": [x0, y0, x1, y1],
            "truncated": int(x0 == 0 or y0 == 0 or x1 == W - 1 or y1 == H - 1),
        }
    return out


def table_check_mm(cam, S, D, N):
    if N is None:
        return None
    m = (S[..., 1] == 1) & (N[..., 1] > 0.99) & (D > 0)
    if m.sum() < 100:
        return None
    pw = cam.unproject_depth(D)
    h = pw[..., 1][m]
    return [round(1000.0 * float(h.mean()), 3), round(1000.0 * float(h.std()), 3)]


def jfloat(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- per image
def process_image(exr_path, scene, grasp, args, nogrip_stats=None):
    S, D, N, W2C, W2N = load_exr(exr_path)
    H, W = D.shape
    cam = Camera(W2C, W2N, W, H)
    m = re.search(r"v(\d+)_g(\d+)\.exr$", os.path.basename(exr_path))
    view_id, with_g = int(m.group(1)), int(m.group(2))

    # view recorded by camera_views (for comparison)
    view = None
    for v in scene.get("views") or []:
        if int(v.get("view_id", -1)) == view_id:
            view = v
    fx_diff = None if view is None or view.get("fx") is None else cam.fx - float(view["fx"])

    # objects
    st = object_stats(S)
    objs_by_id = {int(o["instance_id"]): o for o in scene.get("objects", [])}
    objects = []
    class_mismatch = 0
    for iid in sorted(st):
        o = dict(st[iid])
        so = objs_by_id.get(iid, {})
        o["instance_id"] = iid
        o["class"] = so.get("class")
        o["name"] = so.get("name")
        if so.get("class_id") is not None and int(so["class_id"]) != o["class_id"]:
            class_mismatch += 1
        if with_g == 1 and nogrip_stats is not None:
            full = nogrip_stats.get(iid, {}).get("pixels", 0)
            o["pixels_without_gripper"] = full
            o["occluded_by_gripper_px"] = max(0, full - o["pixels"])
            o["visible_frac_vs_no_gripper"] = round(o["pixels"] / full, 4) if full else None
        objects.append(o)

    tm = S[..., 1] == 2
    pv, pc = np.unique(S[..., 2][tm], return_counts=True)
    names = {v: k for k, v in (scene.get("part_ids") or {}).items()}
    target_parts = {names.get(int(a), str(int(a))): int(b) for a, b in zip(pv, pc)}

    # grasp keypoints
    kps = []
    on_mask = None
    if grasp is not None:
        yaw = float(scene.get("target_yaw") or 0.0)
        T = scene.get("target_pos") or [0, 0, 0]
        cpos = as_points(grasp.get("c_pos"))
        chains = grasp.get("c_chain") or [""] * len(cpos)
        zones = grasp.get("c_zone") or [""] * len(cpos)
        pts = [("contact", chains[i] if i < len(chains) else "", zones[i] if i < len(zones) else "", cpos[i])
               for i in range(len(cpos))]
        gp = grasp.get("gripper_pose") or {}
        if gp.get("pos") is not None:
            pts.append(("wrist_connect", "", "", np.asarray(gp["pos"], dtype=np.float64)))
        if pts:
            pm = np.array([p[3] for p in pts])
            pw = mug_to_world(pm, yaw, T)
            u, v, z = cam.project(pw)
            n_c, n_ok = 0, 0
            for i, (kind, chain, zone, _) in enumerate(pts):
                ui, vi = int(round(u[i])), int(round(v[i]))
                inside = bool(z[i] > 0 and 0 <= ui < W and 0 <= vi < H)
                rec = {"kind": kind, "chain": chain, "zone": zone,
                       "world": [jfloat(x) for x in pw[i]],
                       "uv": [jfloat(u[i], 2), jfloat(v[i], 2)], "z": jfloat(z[i]),
                       "in_image": int(inside)}
                if inside:
                    dpx = D[vi, ui]
                    rec["depth_at_px"] = jfloat(dpx)
                    rec["visible"] = int(dpx > 0 and abs(dpx - z[i]) < args.vis_tol)
                    rec["mask_class"] = int(S[vi, ui, 1])
                    rec["mask_instance"] = int(S[vi, ui, 0])
                    if kind == "contact":
                        n_c += 1
                        # the contact lies between finger and mug: in _g1 it lands on the mug or gripper,
                        # in _g0 on the mug (or something in front of it)
                        n_ok += int(rec["mask_class"] in ((2, 9) if with_g else (2,)) or
                                    (rec["mask_class"] > 0 and dpx < z[i] - args.vis_tol))
                else:
                    rec["visible"] = 0
                kps.append(rec)
            on_mask = round(n_ok / n_c, 3) if n_c else None

    meta = {
        "image_metadata_version": VERSION,
        "scene_id": scene.get("scene_id"),
        "view_id": view_id,
        "with_gripper": with_g,
        "exr": os.path.basename(exr_path),
        "pair": os.path.basename(exr_path).replace("_g%d" % with_g, "_g%d" % (1 - with_g)),
        "target_mug_id": scene.get("target_mug_id"),
        "grasp_sample_id": (scene.get("grasp") or {}).get("sample_id"),
        "grasp_final_tasks": (scene.get("grasp") or {}).get("final_tasks"),
        "camera": {
            "width": W, "height": H,
            "K_opencv": [[jfloat(x) for x in r] for r in cam.K()],
            "world_to_cam_opencv": [[jfloat(x) for x in r] for r in cam.world_to_cam_cv],
            "cam_to_world_usd": [[jfloat(x) for x in r] for r in cam.cam_to_world_usd],
            "conventions": "Houdini world (Y up, metres); K and world_to_cam in OpenCV "
                           "(x right, y down, z forward, centre of pixel (0,0) at (0,0)); "
                           "cam_to_world_usd with -Z forward",
            "depth": "camera Z in metres (ray:hitPz, nearest sample); 0 = background",
            "fstop": view.get("fstop") if view else None,
            "focus_dist": view.get("focus_dist") if view else None,
            "hfov_deg": view.get("hfov_deg") if view else None,
        },
        "objects": objects,
        "target_parts_px": target_parts,
        "grasp_keypoints": kps,
        "checks": {
            "fx_header_minus_camera_views_px": jfloat(fx_diff, 4),
            "principal_offset_ndc": [jfloat(cam.ox, 6), jfloat(cam.oy, 6)],
            "table_top_height_mm_mean_std": table_check_mm(cam, S, D, N),
            "contacts_on_expected_mask_frac": on_mask,
            "instances_with_class_mismatch": class_mismatch,
            "background_px_with_depth": int(((S[..., 1] == 0) & (D > 0)).sum()),
        },
    }
    return meta, st


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip", required=True)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--grasp_dir", default="")
    ap.add_argument("--vis_tol", type=float, default=0.01)
    ap.add_argument("--scene_min", type=int, default=-1)
    ap.add_argument("--scene_max", type=int, default=-1)
    ap.add_argument("--summary", default="metadata_summary.csv")
    args = ap.parse_args()

    rdir = os.path.join(args.hip, "renders")
    sdirs = sorted(glob.glob(os.path.join(rdir, "scene_*")))
    if args.scenes:
        want = {int(s) for s in args.scenes.split(",") if s.strip()}
        sdirs = [d for d in sdirs if int(os.path.basename(d)[6:]) in want]
    if args.scene_min >= 0:
        sdirs = [d for d in sdirs if int(os.path.basename(d)[6:]) >= args.scene_min]
    if args.scene_max >= 0:
        sdirs = [d for d in sdirs if int(os.path.basename(d)[6:]) <= args.scene_max]
    print("scenes to process: %d" % len(sdirs))
    rows = []
    n_img = 0
    for sd in sdirs:
        sid = int(os.path.basename(sd)[6:])
        sj = os.path.join(args.hip, "scenes", "scene_%06d" % sid, "scene.json")
        if not os.path.isfile(sj):
            print("[warning] missing %s" % sj)
            continue
        with open(sj, encoding="utf-8") as fh:
            scene = json.load(fh)
        grasp = None
        gid = (scene.get("grasp") or {}).get("sample_id")
        if gid is not None:
            gp = find_grasp_json(args.hip, args.grasp_dir, gid)
            if gp:
                with open(gp, encoding="utf-8") as fh:
                    grasp = json.load(fh)
            else:
                print("[warning] scene %d: sample_%07d.json not found" % (sid, int(gid)))
        exrs = sorted(glob.glob(os.path.join(sd, "v*_g*.exr")))
        views = sorted({re.search(r"v(\d+)_g", os.path.basename(e)).group(1) for e in exrs})
        for v in views:
            st0 = None
            for g in (0, 1):                       # without gripper first: gives the full pixel counts
                ep = os.path.join(sd, "v%s_g%d.exr" % (v, g))
                if not os.path.isfile(ep):
                    print("[warning] missing %s" % ep)
                    continue
                meta, st = process_image(ep, scene, grasp, args, st0 if g == 1 else None)
                if g == 0:
                    st0 = st
                out = ep[:-4] + ".json"
                with open(out + ".tmp", "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=1, ensure_ascii=False)
                os.replace(out + ".tmp", out)
                n_img += 1
                tgt = [o for o in meta["objects"] if o["class_id"] == 2]
                ck = meta["checks"]
                rows.append({
                    "scene_id": sid, "view_id": int(v), "with_gripper": g,
                    "target_px": tgt[0]["pixels"] if tgt else 0,
                    "target_bbox": " ".join(map(str, tgt[0]["bbox_xyxy"])) if tgt else "",
                    "target_truncated": tgt[0]["truncated"] if tgt else "",
                    "target_vis_frac": tgt[0].get("visible_frac_vs_no_gripper", "") if tgt else "",
                    "handle_px": meta["target_parts_px"].get("handle", 0),
                    "n_objects": len(meta["objects"]),
                    "contacts_on_mask": ck["contacts_on_expected_mask_frac"],
                    "fx_diff_px": ck["fx_header_minus_camera_views_px"],
                    "table_mm": ck["table_top_height_mm_mean_std"][0] if ck["table_top_height_mm_mean_std"] else "",
                    "class_mismatch": ck["instances_with_class_mismatch"],
                    "bg_depth_px": ck["background_px_with_depth"],
                })
        print("scene %06d: %d views" % (sid, len(views)))

    if rows:
        out = os.path.join(rdir, args.summary)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        def col(k, g=None):
            return [r[k] for r in rows if r[k] not in ("", None) and (g is None or r["with_gripper"] == g)]
        print("\n=== summary (%d images) ===" % n_img)
        tp = col("target_px")
        print("target mug: median px %d, images without visible mug %d" %
              (int(np.median(tp)), sum(1 for x in tp if x == 0)))
        print("mug truncated (touches border): %d" % sum(1 for r in rows if r["target_truncated"] == 1))
        vf = col("target_vis_frac", 1)
        if vf:
            print("mug visibility with gripper (vs without): median %.2f, <0.25: %d of %d" %
                  (float(np.median(vf)), sum(1 for x in vf if x < 0.25), len(vf)))
        print("handle visible (no gripper): %d of %d" % (sum(1 for r in rows if r["with_gripper"] == 0 and r["handle_px"] > 0),
                                                    sum(1 for r in rows if r["with_gripper"] == 0)))
        cm = col("contacts_on_mask")
        if cm:
            print("contacts on the expected mask: mean %.3f, images < 0.5: %d" %
                  (float(np.mean(cm)), sum(1 for x in cm if x < 0.5)))
        fd = col("fx_diff_px")
        if fd:
            print("fx header - camera_views: max |diff| %.4f px" % max(abs(x) for x in fd))
        tm_ = col("table_mm")
        if tm_:
            print("table top: mean %.3f mm (range %.3f .. %.3f)" % (np.mean(tm_), min(tm_), max(tm_)))
        print("instances with inconsistent class: %d   background pixels with depth: %d" %
              (sum(r["class_mismatch"] for r in rows), sum(r["bg_depth_px"] for r in rows)))
        print("CSV: %s" % out)
    else:
        print("No image was processed (is the --hip path right?)")


if __name__ == "__main__":
    main()
