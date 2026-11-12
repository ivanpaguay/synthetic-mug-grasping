# =========================================================================
# family_run  v1.1   (Python 3, standard library only)
#
# Validates a whole mug FAMILY in Isaac Sim: runs grasp_physics_test.py once
# per mug (one Isaac process per mug) and finally merges everything into one
# isaac_results_<tag>.json for merge_physics_results.py.
#
# Why one process per mug instead of all mugs in one:
#   - each mug has its own USD, mass and SDF collision (cooked); changing
#     them once the simulation exists is exactly what crashed Isaac early on;
#   - if a mug fails or Isaac crashes, only that mug is lost and the rest go on;
#   - runs can be resumed: finished mugs are skipped.
#   Isaac start-up (~20-40 s) is small next to 10-30 graded grasps.
#
# Frozen configuration (ALWAYS passed to every run):
#   --force_levels "1,2,4,8,16" --mug_collision sdf --finger_collision convexDecomposition
#   Anything after "--" is appended at the end and wins (e.g. --mu 0.3).
#
# Usage (from the Isaac Sim folder, with its python.bat):
#   python.bat family_run.py
#       --script isaac/grasp_physics_test.py --robot <robotiq_3f>.usda
#       --data   <SYNTH_ROOT>/HIPS/grasp_data_E  --out <SYNTH_ROOT>/HIPS/isaac_family_E
#       --tag fam_E  [--mugs 100,101]  [--rerun]  [--timeout_min 60]  [-- --mu 0.3]
#
# Output in --out:
#   per_mug_<tag>/isaac_results_mug_XXXX.json/.csv   result of each mug
#   per_mug_<tag>/log_mug_XXXX.txt                   grasp_physics_test log
#   per_mug_<tag>/console_mug_XXXX.txt               full Isaac console
#   isaac_results_<tag>.json/.csv                    ALL mugs together
#   family_run_<tag>.txt                             summary of this run
#   (per_mug is one folder per tag, so a new run with another tag does not
#    skip mugs finished in a different run)
# =========================================================================
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

VERSION = "v1.1"
FROZEN = ["--force_levels", "1,2,4,8,16", "--mug_collision", "sdf",
          "--finger_collision", "convexDecomposition"]
NO_SAMPLES = 3                       # exit code of grasp_physics_test when there are no samples


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="path to grasp_physics_test.py")
    ap.add_argument("--robot", required=True, help="gripper USD")
    ap.add_argument("--data", required=True, help="folder with the sample_XXXXXXX.json files")
    ap.add_argument("--out", required=True, help="results folder")
    ap.add_argument("--tag", required=True, help="run name (e.g. fam_E)")
    ap.add_argument("--hip", default="", help="folder that contains mugs/ (default: parent of --data)")
    ap.add_argument("--mugs", default="", help="only these mugs, e.g. 100,101 (default: all)")
    ap.add_argument("--rerun", action="store_true", help="also repeat mugs already finished")
    ap.add_argument("--timeout_min", type=float, default=0, help="time limit per mug (0 = none)")
    ap.add_argument("--python", default="", help="Isaac python.bat (default: the one in the current folder)")
    args = ap.parse_args(argv)

    py = args.python or (os.path.abspath("python.bat") if os.path.isfile("python.bat") else sys.executable)
    per = os.path.join(args.out, f"per_mug_{args.tag}")
    os.makedirs(per, exist_ok=True)
    summary_path = os.path.join(args.out, f"family_run_{args.tag}.txt")
    summ = open(summary_path, "a", encoding="utf-8")

    def say(msg):
        line = f"[family_run] {msg}"
        print(line, flush=True)
        summ.write(line + "\n")
        summ.flush()

    # mugs present in the data (7-digit names: MMMM GGG)
    mugs = {}
    for f in glob.glob(os.path.join(args.data, "sample_*.json")):
        m = re.fullmatch(r"sample_(\d{7})\.json", os.path.basename(f))
        if m:
            mid = int(m.group(1)) // 1000
            mugs[mid] = mugs.get(mid, 0) + 1
    if not mugs:
        say(f"ERROR: no sample_XXXXXXX.json (7 digits) in {args.data}")
        sys.exit(1)
    wanted = sorted(mugs)
    if args.mugs:
        sel = {int(x) for x in args.mugs.split(",") if x.strip()}
        missing = sorted(sel - set(mugs))
        if missing:
            say(f"WARNING: these mugs have no samples: {missing}")
        wanted = [m for m in wanted if m in sel]

    say(f"version {VERSION}  tag={args.tag}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"Isaac python: {py}")
    say(f"mugs: {len(wanted)}  samples (all labels): {sum(mugs[m] for m in wanted)}")
    say(f"configuration: {' '.join(FROZEN + extra)}")

    ok, failed, empty, skipped = [], [], [], []
    t_start = time.time()
    done_times = []
    for i, mid in enumerate(wanted, 1):
        res = os.path.join(per, f"isaac_results_mug_{mid:04d}.json")
        if os.path.isfile(res) and not args.rerun:
            skipped.append(mid)
            ok.append(mid)
            say(f"[{i}/{len(wanted)}] mug {mid:4d}: already finished, skipped (--rerun to repeat)")
            continue
        if os.path.isfile(res):
            os.remove(res)
        cmd = [py, args.script, "--robot", args.robot, "--data", args.data, "--out", per,
               "--mug_id", str(mid), "--tag", args.tag, "--result_file", res] + \
              (["--hip", args.hip] if args.hip else []) + FROZEN + extra
        console = os.path.join(per, f"console_mug_{mid:04d}.txt")
        t0 = time.time()
        rc = None
        with open(console, "w", encoding="utf-8", errors="replace") as fh:
            try:
                p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=(args.timeout_min * 60) or None)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        dt = time.time() - t0
        if os.path.isfile(res):
            # the result exists -> the mug finished (even if Isaac returns an odd code on exit)
            with open(res, encoding="utf-8") as fh:
                rows = json.load(fh).get("results", [])
            n6 = sum(int(r.get("held_count") == 6) for r in rows)
            ok.append(mid)
            done_times.append(dt)
            say(f"[{i}/{len(wanted)}] mug {mid:4d}: {len(rows)} samples, {n6} hold 6/6  ({dt:.0f} s)")
        elif rc == NO_SAMPLES:
            empty.append(mid)
            say(f"[{i}/{len(wanted)}] mug {mid:4d}: no samples to test (none passes the filter)  ({dt:.0f} s)")
        else:
            failed.append(mid)
            say(f"[{i}/{len(wanted)}] mug {mid:4d}: FAILED (code {rc}, {dt:.0f} s) -> see {console}")
        if done_times:
            left = len(wanted) - i
            eta = left * sum(done_times) / len(done_times)
            say(f"    elapsed {(time.time() - t_start) / 60:.1f} min, ~{eta / 60:.1f} min left")

    # ---------------- merge all mugs into one file
    all_rows, cfg = [], None
    for mid in ok:
        with open(os.path.join(per, f"isaac_results_mug_{mid:04d}.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        if cfg is None:
            cfg = dict(d.get("config", {}))
        all_rows.extend(d.get("results", []))
    all_rows.sort(key=lambda r: int(r["sample_id"]))
    if cfg is not None:
        # what changes per mug (USD, mass, centre of mass) is already in mug.json and in each row
        for k in ("mug_id", "mug_id_used", "mug_usd", "mass_used", "mass_source", "com_used", "result_file"):
            cfg.pop(k, None)
        cfg["tag"] = args.tag
        cfg["family"] = {"run_version": VERSION, "mugs_ok": ok, "mugs_failed": failed,
                         "mugs_no_samples": empty}
        out_json = os.path.join(args.out, f"isaac_results_{args.tag}.json")
        with open(out_json + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"config": cfg, "results": all_rows}, fh, indent=1)
        os.replace(out_json + ".tmp", out_json)
        if all_rows:
            keys = []
            for r in all_rows:
                keys += [k for k in r if k not in keys]
            with open(os.path.splitext(out_json)[0] + ".csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(all_rows)
        say(f"merged: {len(all_rows)} samples from {len(ok)} mugs -> {out_json}")
    say(f"finished {len(ok)} (skipped {len(skipped)}), no samples {len(empty)}, failed {len(failed)}"
        + (f": {failed}" if failed else ""))
    say(f"total time: {(time.time() - t_start) / 60:.1f} min")
    summ.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
