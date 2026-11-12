# =========================================================================
# final_table  v1   (any Python 3, no dependencies)
# Collects the metrics.json files of the final evaluation (one folder per
# model under --dir) into a table: per model and per configuration (mean of
# its seeds). Configuration = folder name without the seed suffix
#   (E_b0 and E_b0_seed1 -> E_b0;  R_b2_s0 and R_b2_s1 -> R_b2)
#
# Usage:
#   python final_table.py --dir <SYNTH_ROOT>/train/final
# Also writes final_table.csv inside --dir.
# =========================================================================
import argparse
import csv
import glob
import json
import os
import re

COLS = [("mIoU", lambda r: r["miou_parts"]), ("handle", lambda r: r["iou"]["handle"]),
        ("mug", lambda r: r["iou_mug"]), ("exterior", lambda r: r["iou_exterior"]),
        ("body", lambda r: r["iou"]["body"]), ("rim", lambda r: r["iou"]["rim"]),
        ("interior", lambda r: r["iou"]["interior"]), ("base", lambda r: r["iou"]["base"])]


def config_of(name):
    return re.sub(r"(_seed\d+|_s\d+)$", "", name)


def num(v):
    return None if v is None or v != v else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    runs = {}
    for mj in sorted(glob.glob(os.path.join(a.dir, "*", "metrics.json"))):
        name = os.path.basename(os.path.dirname(mj))
        with open(mj, encoding="utf-8") as fh:
            m = json.load(fh)
        key = [k for k in m if k.startswith("real_")]
        if not key:
            continue
        runs[name] = m[key[0]]
    if not runs:
        raise SystemExit("no metrics.json with real results found in %s" % a.dir)
    groups = []
    for res in runs.values():
        for g in res:
            if g not in groups:
                groups.append(g)
    order = ["main", "in_family", "wide_angle", "all"]
    groups = [g for g in order if g in groups] + sorted(g for g in groups if g not in order)

    rows = []
    for name, res in runs.items():
        for g in groups:
            for variant in ("band", "strict"):
                r = res.get(g, {}).get(variant)
                if not r:
                    continue
                row = {"model": name, "config": config_of(name), "group": g, "variant": variant}
                for c, f in COLS:
                    row[c] = num(f(r))
                rows.append(row)

    def show(title, g, variant, by_config):
        sel = [r for r in rows if r["group"] == g and r["variant"] == variant]
        if not sel:
            return
        print("\n=== %s ===" % title)
        print("%-16s %3s " % ("config" if by_config else "model", "n") + " ".join("%8s" % c for c, _ in COLS))
        if by_config:
            confs = []
            for r in sel:
                if r["config"] not in confs:
                    confs.append(r["config"])
            pref = ["pilot_b0", "E_b0", "E_b2", "R_b2", "FT_b2"]
            confs.sort(key=lambda c: (pref.index(c) if c in pref else len(pref), c))
            for cf in confs:
                rr = [r for r in sel if r["config"] == cf]
                vals = []
                for c, _ in COLS:
                    v = [r[c] for r in rr if r[c] is not None]
                    if not v:
                        vals.append("     -  ")
                    elif len(v) == 1:
                        vals.append("%8.3f" % v[0])
                    else:
                        vals.append("%8.3f" % (sum(v) / len(v)))
                rng = [r["mIoU"] for r in rr if r["mIoU"] is not None]
                extra = "   (mIoU %s)" % " / ".join("%.3f" % x for x in rng) if len(rng) > 1 else ""
                print("%-16s %3d " % (cf, len(rr)) + " ".join(vals) + extra)
        else:
            for r in sel:
                print("%-16s %3s " % (r["model"], "") +
                      " ".join("%8.3f" % r[c] if r[c] is not None else "     -  " for c, _ in COLS))

    show("FINAL TEST - main (no wide angle) - 2 px band - mean of seeds", "main", "band", True)
    show("FINAL TEST - main - strict - mean of seeds", "main", "strict", True)
    show("FINAL TEST - wide angle (out of distribution) - 2 px band", "wide_angle", "band", True)
    show("FINAL TEST - main - 2 px band - each model", "main", "band", False)
    for g in groups:
        if g in ("main", "all", "wide_angle"):
            continue
        show("group %s - 2 px band" % g, g, "band", True)

    out = os.path.join(a.dir, "final_table.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nCSV: %s" % out)


if __name__ == "__main__":
    main()
