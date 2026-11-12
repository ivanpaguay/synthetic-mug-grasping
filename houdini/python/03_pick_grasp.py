# =========================================================================
# pick_grasp  v1   (Python SOP, no inputs)
# Houdini node: /obj/geo5/pick_grasp
# Picks the grasp for the WITH-GRIPPER variant of a scene.
#
# Reads the grasp index written by isaac/merge_physics_results.py and picks,
# reproducibly (from scene_id), a grasp of the target mug whose final label is
# valid. The mug and the scene come from the parameters of the ../scene_layout
# node (target_mug_id, scene_id), so they are not duplicated here.
#
# OPTIONAL parameters (defaults are used if they do not exist):
#   index_csv     (string) $HIP/grasp_data_E_physics/dataset_index.csv
#   grasp_geo_dir (string) $HIP/grasp_data_E/geo
#   require_task  (string) e.g. pour: only grasps with that task ("" = any)
#   layout_node   (string) ../scene_layout
#
# Output: empty geometry with detail attributes:
#   grasp_sample_id (-1 if there is none), grasp_geo_path, grasp_final_tasks,
#   grasp_mode, grasp_candidates (how many valid ones there were), pick_grasp_version
# =========================================================================
import csv
import os
import random

node = hou.pwd()
geo = node.geometry()


def parm_or(name, default):
    p = node.parm(name)
    if p is None:
        return default
    v = p.evalAsString().strip()
    return v if v else default


hip = hou.text.expandString("$HIP")
index_csv = parm_or("index_csv", hip + "/grasp_data_E_physics/dataset_index.csv")
geo_dir = parm_or("grasp_geo_dir", hip + "/grasp_data_E/geo")
require = parm_or("require_task", "")
layout = hou.node(parm_or("layout_node", "../scene_layout"))
if layout is None:
    raise hou.NodeError("pick_grasp: scene_layout node not found (layout_node parameter)")

mug_id = int(layout.parm("target_mug_id").eval())
scene_id = int(layout.parm("scene_id").eval())

if not os.path.isfile(index_csv):
    raise hou.NodeError("pick_grasp: index not found: %s" % index_csv)

cands = []
with open(index_csv, encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        if str(r.get("mug_id")) != str(mug_id):
            continue
        if r.get("final_label") not in ("valid", "valid_weak"):
            continue
        tasks = (r.get("final_tasks") or "").split()
        if require and require not in tasks:
            continue
        cands.append(r)
cands.sort(key=lambda r: int(r["sample_id"]))

for name, default in (("grasp_sample_id", -1), ("grasp_candidates", 0)):
    geo.addAttrib(hou.attribType.Global, name, default)
for name in ("grasp_geo_path", "grasp_final_tasks", "grasp_mode", "pick_grasp_version"):
    geo.addAttrib(hou.attribType.Global, name, "")

geo.setGlobalAttribValue("grasp_candidates", len(cands))
geo.setGlobalAttribValue("pick_grasp_version", "v1")
if not cands:
    node.addWarning("pick_grasp: mug %d has no valid grasps%s"
                    % (mug_id, (" with the task '%s'" % require) if require else ""))
else:
    r = random.Random(scene_id * 7919 + mug_id).choice(cands)
    sid = int(r["sample_id"])
    path = geo_dir.rstrip("/") + "/sample_%07d.bgeo.sc" % sid      # Houdini uses / on Windows too
    if not os.path.isfile(path):
        node.addWarning("pick_grasp: file not found: %s" % path)
    geo.setGlobalAttribValue("grasp_sample_id", sid)
    geo.setGlobalAttribValue("grasp_geo_path", path)
    geo.setGlobalAttribValue("grasp_final_tasks", r.get("final_tasks") or "")
    geo.setGlobalAttribValue("grasp_mode", r.get("grasp_mode") or "")
