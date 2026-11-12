# =========================================================================
# scene_json  v1.4   (Python SOP)
# Houdini node: /obj/geo5/scene_json      in0 = semantic_ids
# Writes the full record of the scene:
#   $HIP/scenes/scene_XXXXXX/scene.json
# and passes the geometry through unchanged.
#
# in0 = final SOP scene (after semantic_ids), with i@instance_id, s@class,
#       i@class_id, s@name, v@Cd, f@roughness, f@metalness, f@transmission,
#       s@mat_type on the primitives.
# Also reads (by path, like pick_grasp):
#   ../scene_layout   layout points (pose, dims, ref_mug_id) + detail
#   ../scene_params   style, HDRI, dome light
#   ../pick_grasp     grasp of the with-gripper variant
#   ../camera_views   camera views: pose, intrinsics (fx, fy, cx, cy, focal,
#                     aperture), depth of field; resolution
# OPTIONAL parameters: out_dir (string, default $HIP/scenes),
#   with_gripper (int, 1 if this scene includes the gripper; default 1 if the
#   geometry has class == "gripper")
# v1.4: class_ids table with the extra distractor classes (10-14)
# v1.3: stores tex_tile per object (print from the UDIM library; -1 = plain)
# v1.2: includes the camera views
# v1.1: ref_mug_id only on mugs (it used to come out as 0 on table, boxes, etc.)
# =========================================================================
import json
import os

VERSION = "v1.4"
node = hou.pwd()
geo = node.geometry()


def parm_or(name, default):
    p = node.parm(name)
    if p is None:
        return default
    v = p.evalAsString().strip()
    return v if v else default


def det(g, name, default=None):
    if g is None or g.findGlobalAttrib(name) is None:
        return default
    v = g.attribValue(name)
    return list(v) if isinstance(v, tuple) else v


def sop_geo(path):
    n = hou.node(path)
    return n.geometry() if n is not None else None


lay = sop_geo(parm_or("layout_node", "../scene_layout"))
par = sop_geo(parm_or("params_node", "../scene_params"))
pk = sop_geo(parm_or("grasp_node", "../pick_grasp"))
if lay is None:
    raise hou.NodeError("scene_json: scene_layout not found")

scene_id = int(det(lay, "scene_id", -1))

# ---------- objects: layout (points) + appearance (primitives) ----------
layout_pts = {}
names = [a.name() for a in lay.pointAttribs()]
for pt in lay.points():
    inst = pt.attribValue("instance_id")
    rec = {}
    for an in ("class", "is_target", "yaw", "radius", "dims", "ref_mug_id", "obj_seed"):
        if an in names:
            v = pt.attribValue(an)
            rec[an] = list(v) if isinstance(v, tuple) else v
    p = pt.position()
    rec["position"] = [p[0], p[1], p[2]]
    layout_pts[inst] = rec

prim_names = [a.name() for a in geo.primAttribs()]
objects = {}
for pr in geo.prims():
    inst = pr.attribValue("instance_id") if "instance_id" in prim_names else -1
    if inst in objects:
        continue
    o = {"instance_id": inst}
    for an in ("class", "class_id", "name", "mat_type", "roughness", "metalness", "transmission", "ref_mug_id",
               "tex_tile"):
        if an in prim_names:
            o[an] = pr.attribValue(an)
    if "Cd" in prim_names:
        o["base_color"] = list(pr.attribValue("Cd"))
    o.update({k: v for k, v in layout_pts.get(inst, {}).items() if k not in o})
    if o.get("class") not in ("mug", "target_mug"):
        o.pop("ref_mug_id", None)          # only mugs point to a mug of the family
    objects[inst] = o

# ---------- camera views ----------
cv = sop_geo(parm_or("views_node", "../camera_views"))
views = []
if cv is not None:
    vnames = [a.name() for a in cv.pointAttribs()]
    for pt in cv.points():
        v = {}
        for an in vnames:
            val = pt.attribValue(an)
            if isinstance(val, tuple):
                val = list(val)
            v["position" if an == "P" else an] = val
        views.append(v)
    views.sort(key=lambda v: v.get("view_id", 0))

has_gripper = any(o.get("class") == "gripper" for o in objects.values())
wg = node.parm("with_gripper")
with_gripper = int(wg.eval()) if wg is not None else int(has_gripper)

rec = {
    "scene_id": scene_id,
    "scene_json_version": VERSION,
    "hip_file": hou.hipFile.path(),
    "target_mug_id": det(lay, "target_mug_id"),
    "target_pos": det(lay, "target_pos"),
    "target_yaw": det(lay, "target_yaw"),
    "table_size": [det(lay, "table_w"), det(lay, "table_d")],
    "n_distractors": det(lay, "n_distractors"),
    "layout_rejected": det(lay, "layout_rejected"),
    "layout_version": det(lay, "layout_version"),
    "scene_style": det(par, "scene_style"),
    "hdri_index": det(par, "hdri_index"),
    "hdri_file": "hdri/hdri_%02d.exr" % int(det(par, "hdri_index", 0) or 0),
    "dome_rot": det(par, "dome_rot"),
    "dome_exposure": det(par, "dome_exposure"),
    "with_gripper": with_gripper,
    "grasp": {
        "sample_id": det(pk, "grasp_sample_id"),
        "final_tasks": det(pk, "grasp_final_tasks"),
        "grasp_mode": det(pk, "grasp_mode"),
        "candidates": det(pk, "grasp_candidates"),
    },
    "objects": [objects[k] for k in sorted(objects)],
    "resolution": [det(cv, "res_x"), det(cv, "res_y")] if cv is not None else None,
    "camera_model": "realsense_d435_color_like (hfov 60-78 deg)",
    "views": views,
    "class_ids": {"background": 0, "table": 1, "target_mug": 2, "mug": 3, "plate": 4,
                  "bowl": 5, "can": 6, "box": 7, "glass": 8, "gripper": 9,
                  "bottle": 10, "jar": 11, "tumbler": 12, "vase": 13, "pitcher": 14},
    "part_ids": {"none": 0, "body": 1, "handle": 2, "rim": 3, "interior": 4, "base": 5},
}

out_dir = parm_or("out_dir", hou.text.expandString("$HIP") + "/scenes")
folder = "%s/scene_%06d" % (out_dir.rstrip("/"), scene_id)
os.makedirs(folder, exist_ok=True)
path = folder + "/scene.json"
with open(path + ".tmp", "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1, ensure_ascii=False)
os.replace(path + ".tmp", path)
