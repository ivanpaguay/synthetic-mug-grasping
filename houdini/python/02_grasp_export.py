# =========================================================================
# grasp_export  v2   (Python SOP)   input: affordances (+ mug_info)
# Houdini node: /obj/geo5/grasp_export1
#
# Writes one JSON record per grasp sample: labels, metrics, gripper pose and
# joint angles (every detail attribute of the incoming geometry).
#  - output folder: optional string parameter "out_dir"; empty -> $HIP/grasp_data_E
#  - 7-digit names: sample_0103014.json (sample_id = mug_id * 1000 + grasp_num)
#  - warns if mug_id is missing (Isaac needs it to load the right mug)
#  - writes to .tmp first and then renames, so a crash never leaves a truncated JSON
# =========================================================================
import hou
import json
import os

VERSION = "v2"
node = hou.pwd()
geo = node.geometry()

SKIP = {"orientjointsparms"}   # KineFX internal attribute, not data
rec = {}
for a in geo.globalAttribs():
    n = a.name()
    if n in SKIP:
        continue
    v = geo.attribValue(n)
    if isinstance(v, tuple):
        v = list(v)
    rec[n] = v

# gripper pose (the "connect" point) and joint angles
joints = {}
grip = None
has_name = geo.findPointAttrib("name") is not None
has_ja   = geo.findPointAttrib("joint_angle") is not None
has_xf   = geo.findPointAttrib("transform") is not None
for pt in geo.points():
    nm = pt.attribValue("name") if has_name else ""
    if has_ja and nm.endswith(("_01", "_02", "_03")):
        joints[nm] = pt.attribValue("joint_angle")
    if nm == "connect":
        P = pt.position()
        grip = {
            "pos": [P[0], P[1], P[2]],
            "rot3x3": list(pt.attribValue("transform")) if has_xf else None,
        }
rec["joint_angles_deg"] = joints
rec["gripper_pose"] = grip
rec["grasp_export_version"] = VERSION

if "mug_id" not in rec:
    print("[grasp_export] WARNING: mug_id is missing (is mug_info upstream of this node?)")

# output folder
out_dir = ""
p = node.parm("out_dir")
if p is not None:
    out_dir = p.evalAsString().strip()
if not out_dir:
    out_dir = hou.text.expandString("$HIP/grasp_data_E")
os.makedirs(out_dir, exist_ok=True)

sid = int(rec.get("sample_id", -1))
fname = ("sample_%07d.json" % sid) if sid >= 0 else "sample_manual.json"
path = os.path.join(out_dir, fname)
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(rec, fh, indent=1)
os.replace(tmp, path)
