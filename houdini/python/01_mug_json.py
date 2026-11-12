# =========================================================================
# mug_json  v1.1   (Python SOP)
# Houdini node: /obj/geo5/python3      in0 = attribwrangle4 (wall_probe)
# Writes ALL detail attributes of the mug to a JSON (generator parameters +
# mug_metrics measurements) and stores its path in s@mug_json_path. The
# geometry passes through unchanged.
# Output path:
#   - if the node has a spare string parameter "json_path", that one;
#   - otherwise it is built automatically: $HIP/mugs/mug_XXXX/mug.json, with
#     XXXX = the mug_id detail attribute (written by mug_metrics).
# v1.1: the json_path parameter is no longer required.
# =========================================================================
import json
import os

node = hou.pwd()
geo = node.geometry()

pj = node.parm("json_path")
path = pj.evalAsString() if pj is not None else ""
if not path:
    if geo.findGlobalAttrib("mug_id") is None:
        raise hou.NodeError("mug_json: the mug_id detail attribute is missing (is mug_metrics upstream?)")
    mug_id = int(geo.attribValue("mug_id"))
    path = os.path.join(hou.text.expandString("$HIP"), "mugs", "mug_%04d" % mug_id, "mug.json")
path = os.path.normpath(path)


def plain(v):
    """Houdini values -> JSON types."""
    if isinstance(v, (tuple, list)):
        return [plain(x) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    return v


data = {a.name(): plain(geo.attribValue(a)) for a in geo.globalAttribs()}
data["mug_json_version"] = "v1"
data["hip_file"] = hou.hipFile.path()

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=1, ensure_ascii=False, sort_keys=True)

if geo.findGlobalAttrib("mug_json_path") is None:
    geo.addAttrib(hou.attribType.Global, "mug_json_path", "")
geo.setGlobalAttribValue("mug_json_path", path)
