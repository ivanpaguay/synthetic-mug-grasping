# =========================================================================
# label_count  v1   (Python 3, standard library only; Isaac's python.bat works)
# Counts the geometric (Houdini) labels in a folder of grasp samples and builds
# the --labels list for the Isaac test: every label that is NOT a hard
# rejection (valid + soft rejections).
# Soft rejections (physics may overturn them): invalid_no_force_closure,
# invalid_missing_contact.
#
# Usage:  python label_count.py <SYNTH_ROOT>/HIPS/grasp_data_E
# =========================================================================
import collections
import glob
import json
import os
import sys

SOFT = {"invalid_no_force_closure", "invalid_missing_contact"}
folder = sys.argv[1]
cnt = collections.Counter()
by_mug = collections.Counter()
for p in glob.glob(os.path.join(folder, "sample_*.json")):
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    lab = str(d.get("grasp_label") or "no_label")
    cnt[lab] += 1
    by_mug[int(d.get("mug_id", -1))] += 1

tot = sum(cnt.values())
print("samples: %d   mugs: %d   (samples per mug: %d-%d)" %
      (tot, len(by_mug), min(by_mug.values() or [0]), max(by_mug.values() or [0])))
for lab, n in cnt.most_common():
    kind = "hard" if lab.startswith("invalid") and lab not in SOFT else ("soft" if lab in SOFT else "")
    print("  %-32s %5d  %5.1f %%  %s" % (lab, n, 100.0 * n / max(1, tot), kind))
keep = [l for l in cnt if not (l.startswith("invalid") and l not in SOFT)]
n_keep = sum(cnt[l] for l in keep)
print("\nIsaac will test %d of %d (~%.1f h at 38 s per sample)" % (n_keep, tot, n_keep * 38 / 3600.0))
print("--labels " + ",".join(sorted(keep)))
