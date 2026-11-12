# =========================================================================
# merge_physics_results  v1.3   (Python 3, standard library only)
#
# Writes the Isaac Sim physics results back into the grasp JSONs and builds a
# CSV index of the whole grasp set.
#
# For each sample_XXXXXXX.json it writes (in --out, NEVER over the originals):
#   "physics_isaac": { "<run tag>": { config, results } , ... }
#   "physics_label":  "hold" | "weak" | "fail" | "no_contact" | "not_tested"
#   "final_label":    final label (rules below)
#   "final_label_source": "isaac:<tag>" | "houdini"
#
# Physics label (from the primary run, --primary):
#   hold        holds in all 6 directions and hold_margin_g >= --min_margin
#   weak        holds in all 6 directions with a smaller margin
#   fail        lets go in some direction
#   no_contact  lets go and, in some direction, no finger touched the mug
#               (the Houdini grasp never forms in the physics)
#   not_tested  the sample did not go to Isaac
# Final label:
#   tested in Isaac -> valid (hold) | valid_weak (weak) | invalid_physics (fail)
#                      | invalid_no_contact (no_contact)
#   not tested      -> the Houdini label is kept
#
# Default minimum margin: 1 x m*g (the mug holds its own weight in all 6
# directions). One level of difference (1 g vs 2 g) is within the protocol's
# noise, so it does not split labels: the margin is kept as a continuous score.
#
# HARD vs SOFT Houdini rejections:
#   - Hard (table collision, collision on arrival, blocked path...): world
#     constraints that Isaac does NOT simulate (no table, no trajectory). Isaac
#     cannot overturn them: the final label stays Houdini's even if physics holds.
#   - Soft (no_force_closure, missing_contact): Houdini's PREDICTIONS of physics;
#     if Isaac holds, the final label is the physics one.
#   - final_tasks: tasks of the final sample. If the final label is valid, the
#     geometric tasks are used (aff_tasks_geo) or, if missing, Houdini's
#     (aff_tasks). If it is not valid: none.
#   - houdini_physics_agree: 1/0 (only tested samples with a soft rejection or
#     approved by Houdini).
#
# Usage:
#   python merge_physics_results.py --data <SYNTH_ROOT>/HIPS/grasp_data_E
#       --results <SYNTH_ROOT>/HIPS/isaac_family_E/isaac_results_fam_E.json [other.json ...]
#       --out <SYNTH_ROOT>/HIPS/grasp_data_E_physics [--primary fam_E] [--min_margin 1]
#       [--log isaac_log.txt]   (only for old runs without closing shift in the JSON)
# =========================================================================
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

VERSION = "v1.3"
DIRS = ["-Y", "+Y", "+X", "-X", "+Z", "-Z"]


def num(v):
    """CSV text -> number (or the text as is)."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) else f
    except (TypeError, ValueError):
        return v


def load_results(path: str):
    """Reads an isaac_results_*.json (with config) or, failing that, a .csv."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d.get("config", {}), d.get("results", [])
    print(f"[merge] WARNING: {os.path.basename(path)} is a CSV: it has no run configuration")
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [{k: num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return {}, rows


def parse_log_shifts(path: str) -> dict:
    """Mug shift on closing (mm) per sample, read from the Isaac log."""
    out, cur = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.search(r"\s([-+][XYZ]): (?:taza movida en el cierre|mug moved on closing)=\s*([\d.]+) mm", line)
            if m:
                cur.append(float(m.group(2)))
            m = re.search(r"(?:muestra|sample)\s+(\d+)\s", line)   # old (Spanish) and current log format
            if m:
                if cur:
                    out[int(m.group(1))] = cur
                cur = []
    return out


def run_tag(cfg: dict, path: str) -> str:
    t = (cfg or {}).get("tag") or ""
    if t:
        return t
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace("isaac_results_", "")


def physics_block(r: dict, cfg: dict, shifts):
    blk = {
        "held_count": r.get("held_count"),
        "mass_kg": r.get("mass_kg"),
        "held_gravity": r.get("held_gravity"),
        "hold_margin_g": r.get("hold_margin_g"),
        "min_fingers_blocked": r.get("min_fingers_blocked"),
        "held": {d: r.get(f"held_{d}") for d in DIRS},
        "disp_mm": {d: r.get(f"disp_{d}_mm") for d in DIRS},
        "rot_deg": {d: r.get(f"rot_{d}_deg") for d in DIRS},
        "fingers_blocked": {d: r.get(f"blocked_{d}") for d in DIRS},
    }
    if any(r.get(f"maxg_{d}") is not None for d in DIRS):
        blk["maxg"] = {d: r.get(f"maxg_{d}") for d in DIRS}
    # shift on closing: from the JSON (grasp_physics_test v3.0+) or from the log
    sh = [r.get(f"shift_{d}_mm") for d in DIRS if r.get(f"shift_{d}_mm") is not None]
    if not sh and shifts:
        sh = shifts
    if sh:
        blk["close_shift_mm_median"] = round(statistics.median(sh), 3)
        blk["close_shift_mm_max"] = round(max(sh), 3)
    if r.get("sim_seconds") is not None:
        blk["sim_seconds"] = r.get("sim_seconds")
    keep = ["script_version", "mu", "mass", "max_effort", "overclose_deg", "stiffness", "damping", "dt",
            "close_s", "hold_s", "max_disp", "max_rot", "force_levels", "force_scale",
            "mug_collision", "finger_collision", "sdf_res", "swap_ring_index", "tag",
            "com_source", "freeze_close"]
    blk["config"] = {k: cfg[k] for k in keep if k in cfg}
    return blk


def classify(blk: dict, min_margin: float) -> str:
    if blk.get("held_count") != 6:
        # lets go: if, in addition, no finger touched the mug, the Houdini grasp
        # never even formed in the physics
        return "no_contact" if blk.get("min_fingers_blocked") == 0 else "fail"
    m = blk.get("hold_margin_g")
    if m is None:                       # run without the graded test
        return "hold"
    return "hold" if m >= min_margin else "weak"


FINAL = {"hold": "valid", "weak": "valid_weak", "fail": "invalid_physics",
         "no_contact": "invalid_no_contact"}
HOUDINI_OK = {"valid", "valid_robust"}
SOFT = {"invalid_no_force_closure", "invalid_missing_contact"}   # Isaac can overturn them
VALID_FINAL = {"valid", "valid_weak"}


def is_hard(houdini_label):
    """Houdini rejection Isaac cannot overturn (every invalid_* that is not soft)."""
    h = str(houdini_label or "")
    return h.startswith("invalid") and h not in SOFT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with Houdini's sample_XXXXXXX.json files")
    ap.add_argument("--results", required=True, nargs="+", help="isaac_results_*.json (o .csv)")
    ap.add_argument("--out", required=True, help="output folder (enriched copies)")
    ap.add_argument("--primary", default="", help="tag of the run that decides the label (default: the first)")
    ap.add_argument("--min_margin", type=float, default=1.0, help="minimum margin (x m*g) for 'hold'")
    ap.add_argument("--log", nargs="*", default=[], help="Isaac logs (only for runs without closing shift in the JSON)")
    args = ap.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(args.data):
        sys.exit("[merge] --out cannot be the original folder (Houdini JSONs are never overwritten)")
    os.makedirs(args.out, exist_ok=True)

    shifts = {}
    for lp in args.log:
        shifts.update(parse_log_shifts(lp))
        print(f"[merge] log {os.path.basename(lp)}: shifts for {len(shifts)} samples")

    runs = {}                                       # tag -> {sample_id: block}
    order = []
    for rp in args.results:
        cfg, rows = load_results(rp)
        tag = run_tag(cfg, rp)
        if tag in runs:
            sys.exit(f"[merge] two runs with the same tag '{tag}': use different --tag values in Isaac")
        runs[tag] = {int(r["sample_id"]): physics_block(r, cfg, shifts.get(int(r["sample_id"])))
                     for r in rows}
        order.append(tag)
        print(f"[merge] run '{tag}': {len(rows)} samples  ({os.path.basename(rp)})")
    primary = args.primary or order[0]
    if primary not in runs:
        sys.exit(f"[merge] --primary '{primary}' is not among {order}")
    print(f"[merge] primary run: '{primary}'  minimum margin: {args.min_margin:g} x m*g")

    files = sorted(glob.glob(os.path.join(args.data, "sample_*.json")))
    if not files:
        sys.exit(f"[merge] no sample_*.json in {args.data}")
    index_rows = []
    counts = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        sid = int(s.get("sample_id", -1))
        phys = {tag: runs[tag][sid] for tag in order if sid in runs[tag]}
        if phys:
            s["physics_isaac"] = phys
        hl = s.get("grasp_label")
        if sid in runs[primary]:
            pl = classify(runs[primary][sid], args.min_margin)
            s["physics_label"] = pl
            if is_hard(hl):
                s["final_label"] = hl                       # Isaac does not overturn hard rejections
                s["final_label_source"] = "houdini_hard"
                s.pop("houdini_physics_agree", None)
            else:
                s["final_label"] = FINAL[pl]
                s["final_label_source"] = f"isaac:{primary}"
                s["houdini_physics_agree"] = int((hl in HOUDINI_OK) == (FINAL[pl] in VALID_FINAL))
        else:
            pl = "not_tested"
            s["physics_label"] = pl
            s["final_label"] = s.get("grasp_label")
            s["final_label_source"] = "houdini"
        if s["final_label"] in VALID_FINAL:
            if s.get("aff_tasks_geo"):
                s["final_tasks"] = s["aff_tasks_geo"]
            elif s.get("aff_tasks") and s.get("aff_tasks") != "ninguna":   # "ninguna" = none (value kept as in the data)
                s["final_tasks"] = s["aff_tasks"]
            else:
                s["final_tasks"] = "lift"      # without geometric tasks: only lift is guaranteed
        else:
            s["final_tasks"] = "ninguna"
        s["merge_isaac_version"] = VERSION
        with open(os.path.join(args.out, os.path.basename(f)), "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=1, ensure_ascii=False)

        key = (s.get("grasp_label"), s["final_label"])
        counts[key] = counts.get(key, 0) + 1
        p = runs[primary].get(sid, {})
        row = {"sample_id": sid, "mug_id": s.get("mug_id"), "grasp_num": s.get("grasp_num"),
               "grasp_mode": s.get("grasp_mode"),
               "houdini_label": s.get("grasp_label"), "physics_label": pl,
               "final_label": s["final_label"], "final_label_source": s.get("final_label_source"),
               "houdini_physics_agree": s.get("houdini_physics_agree"), "final_tasks": s.get("final_tasks"),
               "eps_mu05": s.get("eps_mu05"), "eps_mu03": s.get("eps_mu03"),
               "n_contacts": s.get("n_contacts"), "approach_rot_deg": s.get("approach_rot_deg"),
               "approach_tilt_deg": s.get("approach_tilt_deg"),
               "grasp_height_ratio": s.get("grasp_height_ratio"),
               "held_count": p.get("held_count"), "hold_margin_g": p.get("hold_margin_g"),
               "close_shift_mm": p.get("close_shift_mm_median"),
               "aff_tasks": s.get("aff_tasks")}
        for tag in order:
            if tag != primary:
                q = runs[tag].get(sid, {})
                row[f"margin_{tag}"] = q.get("hold_margin_g")
                row[f"held_count_{tag}"] = q.get("held_count")
        index_rows.append(row)

    idx_path = os.path.join(args.out, "dataset_index.csv")
    with open(idx_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)

    print(f"[merge] {len(files)} JSON files written to {args.out}")
    print(f"[merge] index: {idx_path}")
    print("[merge] Houdini label -> final label:")
    for (h, fl), n in sorted(counts.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        print(f"[merge]   {str(h):28s} -> {str(fl):20s} {n:4d}")


if __name__ == "__main__":
    main()
