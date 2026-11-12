# =========================================================================
# isaac_clip_candidates  v1   (any Python 3, no dependencies)
# Picks grasps to RECORD in Isaac Sim (--gui) for the reel, from the merged
# physics results (grasp JSONs with physics_isaac.<tag>, e.g. fam_E).
#
# Three lists:
#   SUCCESS   holds in all 6 directions, high margin, the mug barely moves;
#             the best ones of each grasp mode (body, handle, rim...)
#   FAILURE   drops under downward gravity (-Y): clear on video
#   CONTRAST  Houdini called it valid and physics rejects it (or the reverse):
#             the "physics corrects geometry" story
# It also writes ready-to-copy commands (isaac/grasp_physics_test.py --gui).
#
# Usage:
#   python isaac_clip_candidates.py --phys <SYNTH_ROOT>/HIPS/grasp_data_E_physics
#       --out <SYNTH_ROOT>/reel/isaac/candidates.txt
# Options: --tag fam_E, --n 3 (per category / mode),
#   --script, --data, --robot, --clip_out (for the commands; the defaults use
#   %SYNTH_ROOT% and %ROBOT%, expanded by cmd when the commands are pasted)
# =========================================================================
import argparse
import glob
import json
import os
from collections import defaultdict


def load(phys, tag):
    out = []
    for f in sorted(glob.glob(os.path.join(phys, "sample_*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        p = (s.get("physics_isaac") or {}).get(tag)
        if not p:
            continue
        disp = p.get("disp_mm") or {}
        held = p.get("held") or {}
        out.append({
            "sample": int(s["sample_id"]), "mug": int(s.get("mug_id", -1)),
            "mode": s.get("grasp_mode", "?"), "zones": "+".join(s.get("c_zone") or []),
            "houdini": s.get("grasp_label", "?"), "physics": s.get("physics_label", "?"),
            "held_count": int(p.get("held_count", 0)), "margin_g": float(p.get("hold_margin_g", 0) or 0),
            "down": int(held.get("-Y", 0)), "max_disp": max(disp.values()) if disp else 99.0,
        })
    return out


def line(r):
    return ("sample %7d  mug %3d  mode %-7s zones %-14s houdini %-24s physics %-6s %d/6  margin %4.1f g  max disp %6.2f mm" %
            (r["sample"], r["mug"], r["mode"], r["zones"], r["houdini"], r["physics"], r["held_count"],
             r["margin_g"], r["max_disp"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phys", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="fam_E")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--script", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "isaac", "grasp_physics_test.py"))
    ap.add_argument("--data", default=r"%SYNTH_ROOT%\HIPS\grasp_data_E")
    ap.add_argument("--robot", default="%ROBOT%")
    ap.add_argument("--clip_out", default=r"%SYNTH_ROOT%\reel\isaac")
    a = ap.parse_args()

    rows = load(a.phys, a.tag)
    if not rows:
        raise SystemExit("no samples with physics_isaac.%s in %s" % (a.tag, a.phys))
    valid_h = lambda r: r["houdini"].startswith("valid")

    by_mode = defaultdict(list)
    for r in rows:
        if r["held_count"] == 6:
            by_mode[r["mode"]].append(r)
    success = []
    for mode, rr in sorted(by_mode.items()):
        rr.sort(key=lambda r: (-r["margin_g"], r["max_disp"]))
        success += rr[:a.n]
    fail = sorted([r for r in rows if r["down"] == 0], key=lambda r: (r["held_count"], -r["max_disp"]))
    fail_vh = [r for r in fail if valid_h(r)]
    fail = (fail_vh + [r for r in fail if not valid_h(r)])[:2 * a.n]
    contrast_a = [r for r in rows if valid_h(r) and r["held_count"] < 6]
    contrast_b = [r for r in rows if not valid_h(r) and r["held_count"] == 6]
    contrast_a.sort(key=lambda r: r["held_count"])
    contrast_b.sort(key=lambda r: -r["margin_g"])

    out = []
    out.append("samples with physics: %d  |  hold 6/6: %d  |  drop downwards: %d" %
               (len(rows), sum(r["held_count"] == 6 for r in rows), sum(r["down"] == 0 for r in rows)))
    out.append("Houdini valid but physics does not hold 6/6: %d  |  Houdini invalid but holds 6/6: %d" %
               (len(contrast_a), len(contrast_b)))
    out.append("\n=== SUCCESS (best per grasp mode) ===")
    out += [line(r) for r in success]
    out.append("\n=== FAILURE (drops under downward gravity; Houdini-valid ones first) ===")
    out += [line(r) for r in fail]
    out.append("\n=== CONTRAST: Houdini valid, physics rejects it ===")
    out += [line(r) for r in contrast_a[:a.n]]
    out.append("\n=== CONTRAST: Houdini invalid, physics holds it ===")
    out += [line(r) for r in contrast_b[:a.n]]

    out.append("\n=== commands (run from the Isaac Sim folder; first: set ROBOT=<path of the gripper USD>) ===")
    seen = set()
    for r in success + fail + contrast_a[:a.n] + contrast_b[:a.n]:
        if r["sample"] in seen:
            continue
        seen.add(r["sample"])
        out.append("python.bat %s --robot %s --mug_id %d --data %s --out %s --gui --sample %d "
                   "--force_levels 1,2,4,8,16 --mug_collision sdf --finger_collision convexDecomposition" %
                   (a.script, a.robot, r["mug"], a.data, a.clip_out, r["sample"]))
    txt = "\n".join(out)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
        print("\nsaved: %s" % a.out)


if __name__ == "__main__":
    main()
