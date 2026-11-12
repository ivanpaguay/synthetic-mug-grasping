# =========================================================================
# make_render_dir  v1.1   (Python Script TOP, between wedge_grip and render_scene)
# Houdini node: /obj/geo5/topnet_scenes/make_render_dir      input: wedge_grip
# Creates $HIP/renders/scene_XXXXXX before the render: husk does not create
# folders ("Unable to create output directory").
# v1.1: explicit import hou (the Python Script TOP does not import it).
# Evaluate Script During: Cook (In-Process).
# =========================================================================
import os
import hou

sid = work_item.attribValue("scene_id")
if sid is None:
    raise RuntimeError("make_render_dir: the work item has no scene_id (check that the branch comes from make_scene_id)")
folder = os.path.join(hou.text.expandString("$HIP"), "renders", "scene_%06d" % int(sid))
os.makedirs(folder, exist_ok=True)
