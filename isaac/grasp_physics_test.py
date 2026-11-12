# =========================================================================
# grasp_physics_test  v4.3   (Isaac Sim 6.1, standalone script)
#
# Graded hold test of the dataset grasps (GraspGen-style protocol):
#   1. the gripper (imported from the URDF) is FIXED at the origin = palm frame;
#   2. the mug is placed in its pose relative to the palm (from the sample JSON);
#   3. the fingers start from the open pose and close to the Houdini angles plus
#      an over-close (squeeze), with limited joint torque;
#   4. no scene gravity: "gravity" is a force k * m * g on the mug, applied in
#      6 world directions (down, up and 4 lateral), k in --force_levels;
#   5. a direction holds if the mug moves < max_disp and turns < max_rot.
#      maxg_<dir> = highest level held; hold_margin_g = minimum over the 6
#      directions (0 = not even the first level).
#
# Frozen configuration used for the dataset: max torque 3.3 N*m per joint
# (~30 N at the fingertip = the Robotiq 3F's MINIMUM programmable force, so the
# labels are conservative), free closing protocol, always with
# --mug_collision sdf --finger_collision convexDecomposition --force_levels "1,2,4,8,16".
#
# Mug family: each run tests ONE mug (--mug_id):
#   - the mug is found automatically: <hip>/<mug_dir>/mug_phys.usd (mug_dir comes
#     in the sample JSON; --hip defaults to the parent folder of --data);
#   - mass from mug.json (mug_mass_kg) unless --mass is given;
#   - centre of mass from mug.json (mug_com): written into the rigid body, and
#     the test force is applied there (--com_source bbox = box centre instead);
#   - samples are read BEFORE starting Isaac: with nothing to test it exits
#     without opening Isaac (exit code 3);
#   - --result_file: exact path of the results JSON (used by family_run.py).
#
# Details that matter (learned the hard way):
#   - the mug is created at its grasp pose in the USD BEFORE Play (created
#     inside the gripper, the first physics step made it explode);
#   - Articulation/RigidPrim wrappers are created BEFORE Play (creating them
#     while simulating invalidates the physics view);
#   - link inertias are computed by PhysX from the collision (the URDF ones are
#     placeholders); gripper self-collisions are disabled (neighbouring convex
#     hulls overlap near the hinges and push the fingers out of their limits);
#   - un-instancing is done in one Sdf.ChangeBlock, never while traversing;
#   - the mug never sleeps (sleepThreshold 0), friction is forced on all
#     gripper materials, and there is a warm-up after Play so the SDF
#     collision is cooked before the first test.
#
# With --gui the scene is shown in its REAL orientation (Houdini Y-up -> Isaac
# Z-up) and the base is fixed to the world at the grasp pose; without --gui it
# is simulated in the palm frame (faster, same result).
# --pose_check: diagnostic mode (exact Houdini pose, kinematic mug, spheres at
# the Houdini contacts). --look (with --gui): fixed grey background, three
# lights and a fixed camera (--look_cam in the Property panel's numbers,
# rotation order --look_order, default YXZ) plus a countdown, for recording.
#
# Usage (from the Isaac Sim folder):
#   python.bat grasp_physics_test.py --robot <robotiq_3f>.usda --mug_id 103
#       --data <SYNTH_ROOT>/HIPS/grasp_data_E --out <results> [--gui --sample 103014]
#   (whole family: family_run.py, one run per mug)
#   --gui --sample N : one sample, with a window, to look at it.
#   without --gui    : every usable sample (sample_valid = 1), headless.
# =========================================================================
from __future__ import annotations

import argparse
import time
SCRIPT_VERSION = "v4.3"
_T0 = time.time()
import csv
import glob
import json
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--robot", required=True, help="gripper USD (imported from the URDF, fixed base)")
parser.add_argument("--mug", default="", help="mug USD (only for old runs without mug_id)")
parser.add_argument("--mug_id", type=int, default=-1, help="test only the samples of this mug")
parser.add_argument("--hip", default="", help="folder that contains mugs/ (default: parent folder of --data)")
parser.add_argument("--com_source", default="json", choices=["json", "bbox"],
                    help="centre of mass: json = mug_com from mug.json; bbox = obj_center from the sample JSON")
parser.add_argument("--result_file", default="", help="exact path of the results JSON")
parser.add_argument("--data", required=True, help="folder with the sample_XXXXXXX.json files")
parser.add_argument("--out", required=True, help="results folder")
parser.add_argument("--gui", action="store_true", help="with a window (for one sample)")
parser.add_argument("--sample", type=int, default=-1, help="test only this sample")
parser.add_argument("--all", action="store_true", help="test all samples, not only usable ones")
parser.add_argument("--swap_ring_index", action="store_true", help="ring = finger_2 (as the skeleton toggle)")
parser.add_argument("--mass", type=float, default=None,
                    help="mug mass (kg). Default: mug_mass_kg from mug.json (or 0.35 if missing)")
parser.add_argument("--mu", type=float, default=0.5, help="friction (the same as gws_check)")
parser.add_argument("--overclose_deg", type=float, default=5.0, help="over-close on top of the Houdini angles")
parser.add_argument("--max_effort", type=float, default=3.3,
                    help="max torque per joint (N*m). 3.3 ~ 30 N at the tip (Robotiq 3F minimum)")
parser.add_argument("--stiffness", type=float, default=1000.0)
parser.add_argument("--damping", type=float, default=20.0)
parser.add_argument("--dt", type=float, default=1.0 / 240.0)
parser.add_argument("--close_s", type=float, default=0.5, help="closing time (s)")
parser.add_argument("--hold_s", type=float, default=2.0, help="time under 'gravity' per direction (s)")
parser.add_argument("--max_disp", type=float, default=0.02, help="maximum mug displacement (m)")
parser.add_argument("--max_rot", type=float, default=20.0, help="maximum mug rotation (deg)")
parser.add_argument("--pose_check", action="store_true",
                    help="diagnostic: exact Houdini pose, kinematic mug, contacts marked")
parser.add_argument("--keep_inertia", action="store_true",
                    help="use the URDF inertias (placeholders) instead of computing them")
parser.add_argument("--mug_collision", default="convexDecomposition",
                    choices=["convexDecomposition", "sdf", "convexHull"])
parser.add_argument("--finger_collision", default="keep",
                    choices=["keep", "convexDecomposition", "sdf", "convexHull"])
parser.add_argument("--sdf_res", type=int, default=256, help="SDF resolution (when sdf is used)")
parser.add_argument("--force_scale", type=float, default=1.0,
                    help="scales the test force (1 = m*g). Useful to check that the force acts")
parser.add_argument("--warmup", type=int, default=240,
                    help="app updates after Play before testing (collision cooking)")
parser.add_argument("--labels", default="",
                    help="only these Houdini labels, comma-separated (ignores sample_valid)")
parser.add_argument("--force_levels", default="",
                    help='force levels in multiples of m*g, e.g. "1,2,4,8,16" (empty = force_scale only)')
parser.add_argument("--samples", default="", help='comma-separated list of sample ids')
parser.add_argument("--tag", default="", help="suffix for the output files")
parser.add_argument("--freeze_close", action="store_true",
                    help="mug held still while closing; released before the force is applied")
parser.add_argument("--settle_s", type=float, default=0.1,
                    help="settling time without force after releasing the mug (s), --freeze_close only")
parser.add_argument("--connect_y", type=float, default=0.0508, help="connect point in the palm frame (m)")
parser.add_argument("--look", action="store_true", help="with --gui: fixed background, lights and camera for recording")
parser.add_argument("--look_az", type=float, default=35.0, help="camera azimuth (deg, from +X)")
parser.add_argument("--look_el", type=float, default=18.0, help="camera elevation (deg)")
parser.add_argument("--look_dist", type=float, default=0.0, help="camera distance (m; 0 = automatic)")
parser.add_argument("--look_focal", type=float, default=50.0, help="camera focal length (mm, 36 mm aperture)")
parser.add_argument("--look_bg", type=float, default=0.22, help="background grey (0-1)")
parser.add_argument("--look_wait", type=float, default=8.0, help="seconds to wait before the test")
parser.add_argument("--look_cam", default="", help='"tx,ty,tz,rx,ry,rz" (m and degrees, as in the Property panel)')
parser.add_argument("--look_aperture", type=float, default=36.0, help="camera horizontal aperture (mm)")
parser.add_argument("--look_order", default="YXZ", help="rotation order of --look_cam (the Rotate panel's)")
args, _ = parser.parse_known_args()

import builtins
os.makedirs(args.out, exist_ok=True)
_LOG = os.path.join(args.out, ("log_sample_%04d" % args.sample if args.sample >= 0 else
                               ("log_" + args.labels.replace(",", "+") if args.labels else "log_all"))
                    + ("_posecheck" if args.pose_check else "") + ("_graded" if args.force_levels else "")
                    + (f"_f{args.force_scale:g}" if args.force_scale != 1.0 else "")
                    + ("_list" if args.samples else "") + (f"_{args.tag}" if args.tag else "")
                    + (f"_mug{args.mug_id:04d}" if args.mug_id >= 0 else "") + ".txt")
if args.result_file:                    # the log goes next to the results file
    _LOG = os.path.splitext(args.result_file)[0].replace("isaac_results", "log") + ".txt"
    if _LOG == os.path.splitext(args.result_file)[0] + ".txt":
        _LOG = os.path.splitext(args.result_file)[0] + "_log.txt"
    os.makedirs(os.path.dirname(os.path.abspath(_LOG)), exist_ok=True)
open(_LOG, "w", encoding="utf-8").close()


def print(*a, **k):                     # noqa: A001  (copied to the log)
    builtins.print(*a, **k)
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(" ".join(str(x) for x in a) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------- samples
# (read BEFORE starting Isaac: with nothing to test, Isaac is not opened)
import sys

_files = []
if args.mug_id >= 0:
    # 7-digit names (grasp_export v2): sample_MMMMGGG.json, MMMM = mug_id
    _files = sorted(glob.glob(os.path.join(args.data, "sample_%04d???.json" % args.mug_id)))
files = _files or sorted(glob.glob(os.path.join(args.data, "sample_*.json")))
samples = [json.load(open(f, encoding="utf-8")) for f in files]
if args.mug_id >= 0:
    samples = [s for s in samples if s.get("mug_id") is not None and int(s["mug_id"]) == args.mug_id]
if args.sample >= 0:
    samples = [s for s in samples if int(s.get("sample_id", -1)) == args.sample]
elif args.samples:
    _ids = {int(x) for x in args.samples.split(",") if x.strip()}
    samples = [s for s in samples if int(s.get("sample_id", -1)) in _ids]
elif args.labels:
    _labs = {x.strip() for x in args.labels.split(",") if x.strip()}
    samples = [s for s in samples if s.get("grasp_label") in _labs]
elif not args.all:
    samples = [s for s in samples if int(s.get("sample_valid", 0)) == 1]
print(f"[isaac_grasp_test] version {SCRIPT_VERSION}")
print(f"[isaac_grasp_test] samples to test: {len(samples)}"
      + (f" (mug {args.mug_id})" if args.mug_id >= 0 else ""))
if not samples:
    _sub = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(args.data, "**", "sample_*.json"), recursive=True)})
    _msg = (f"No samples to test in {args.data} "
            f"({len(files)} sample_*.json found directly in the folder).")
    if _sub and not files:
        _msg += f" There are sample_*.json files in: {_sub[:3]} -> use that folder as --data."
    elif files:
        _msg += " None passes the filter (--mug_id / --sample / --samples / --labels / sample_valid)."
    print(f"[isaac_grasp_test] NO SAMPLES: {_msg}")
    sys.exit(3)

# ---------------------------------------------------------------- mug
HIP = args.hip or os.path.dirname(os.path.abspath(args.data))
_mids = sorted({s.get("mug_id") for s in samples}, key=lambda v: (v is None, v))
MUG_ID = None
MUG_JSON = {}
if args.mug:
    MUG_USD = args.mug
    if len(_mids) > 1:
        print(f"[isaac_grasp_test] WARNING: --mug is fixed but the samples come from several mugs {_mids}")
else:
    if None in _mids:
        print("[isaac_grasp_test] ERROR: some samples have no mug_id (old JSONs): use --mug MUG.usd")
        sys.exit(1)
    if len(_mids) > 1:
        print(f"[isaac_grasp_test] ERROR: the samples belong to {len(_mids)} mugs {_mids[:10]}: "
              "use --mug_id N or family_run.py (one run per mug)")
        sys.exit(1)
    MUG_ID = int(_mids[0])
    _mdir = samples[0].get("mug_dir") or ("mugs/mug_%04d" % MUG_ID)
    MUG_USD = os.path.normpath(os.path.join(HIP, _mdir, "mug_phys.usd"))
    _mj = os.path.normpath(os.path.join(HIP, _mdir, "mug.json"))
    if os.path.isfile(_mj):
        MUG_JSON = json.load(open(_mj, encoding="utf-8"))
        if MUG_JSON.get("mug_id") is not None and int(MUG_JSON["mug_id"]) != MUG_ID:
            print(f"[isaac_grasp_test] ERROR: {_mj} says mug_id={MUG_JSON['mug_id']}, expected {MUG_ID}")
            sys.exit(1)
    else:
        print(f"[isaac_grasp_test] WARNING: {_mj} does not exist (default mass and centre of mass)")
if not os.path.isfile(MUG_USD):
    print(f"[isaac_grasp_test] ERROR: mug USD not found: {MUG_USD}")
    sys.exit(1)
if args.mass is not None:
    MASS, MASS_SRC = float(args.mass), "--mass"
elif MUG_JSON.get("mug_mass_kg"):
    MASS, MASS_SRC = float(MUG_JSON["mug_mass_kg"]), "mug.json"
else:
    MASS, MASS_SRC = 0.35, "default"
MUG_COM = None                                   # None = obj_center of each sample is used
if args.com_source == "json":
    if MUG_JSON.get("mug_com") is not None:
        MUG_COM = [float(v) for v in MUG_JSON["mug_com"]]
    else:
        print("[isaac_grasp_test] WARNING: no mug_com in mug.json -> box centre (obj_center)")
print(f"[isaac_grasp_test] mug: {MUG_USD}" + (f"  (mug_id {MUG_ID})" if MUG_ID is not None else ""))
print(f"[isaac_grasp_test] mass: {MASS:.4f} kg ({MASS_SRC})  centre of mass: "
      + (f"({MUG_COM[0]:+.4f}, {MUG_COM[1]:+.4f}, {MUG_COM[2]:+.4f}) m (mug.json)" if MUG_COM else "obj_center per sample"))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": not args.gui})

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


# ---------------------------------------------------------------- utilities
def quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) of a rotation matrix (column vectors)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rot_from_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def palm_pose(sample: dict):
    """Palm pose in the Houdini world (R with columns as axes, t)."""
    gp = sample["gripper_pose"]
    rows = np.array(gp["rot3x3"], dtype=float).reshape(3, 3)   # rows = palm axes (Houdini)
    R = rows.T
    t = np.array(gp["pos"], dtype=float) - args.connect_y * rows[1]
    return R, t


def find_prims(root: Usd.Prim, pred):
    return [p for p in Usd.PrimRange(root) if pred(p)]


os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- scene
SimulationManager.set_physics_sim_device("cpu")
simulation_app.update()
stage_utils.create_new_stage()
import omni.usd
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

def add_ref(prim_path: str, file_path: str) -> Usd.Prim:
    """References a USD. If the file has no defaultPrim, its first root prim is used."""
    if not os.path.isfile(file_path):
        raise RuntimeError(f"File not found: {file_path}")
    layer = Sdf.Layer.FindOrOpen(file_path)
    if layer is None:
        raise RuntimeError(f"USD could not open: {file_path}")
    roots = [p.path for p in layer.rootPrims]
    target = None
    if layer.defaultPrim:
        target = Sdf.Path("/" + layer.defaultPrim)
    else:
        # Houdini writes /HoudiniLayerInfo at the root: pick the first root with meshes
        tmp = Usd.Stage.Open(file_path)
        for child in tmp.GetPseudoRoot().GetChildren():
            if child.GetName() == "HoudiniLayerInfo":
                continue
            if any(q.IsA(UsdGeom.Mesh) for q in Usd.PrimRange(child)):
                target = child.GetPath()
                break
        if target is None and roots:
            target = roots[0]
    print(f"[isaac_grasp_test] {os.path.basename(file_path)}: defaultPrim='{layer.defaultPrim}' "
          f"roots={[str(r) for r in roots]} -> using {target}")
    if target is None:
        raise RuntimeError(f"The file has no prims: {file_path}")
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(file_path, target)
    return prim


def gf_matrix(R: np.ndarray, t) -> Gf.Matrix4d:
    """USD matrix (row vectors) from a rotation R (column vectors) and a translation t."""
    return Gf.Matrix4d(float(R[0, 0]), float(R[1, 0]), float(R[2, 0]), 0.0,
                       float(R[0, 1]), float(R[1, 1]), float(R[2, 1]), 0.0,
                       float(R[0, 2]), float(R[1, 2]), float(R[2, 2]), 0.0,
                       float(t[0]), float(t[1]), float(t[2]), 1.0)


# scene: with --gui it is placed at the real grasp pose (see below)
WORLD_VIEW = bool(args.gui)
C_H2I = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # Houdini (Y up) -> Isaac (Z up)
scene_root = stage.DefinePrim("/World/scene", "Xform")
R_S = np.eye(3)
t_S = np.zeros(3)
if WORLD_VIEW and samples:
    _Rp, _tp = palm_pose(samples[0])
    R_S = C_H2I @ _Rp
    t_S = C_H2I @ _tp
    UsdGeom.Xformable(scene_root).AddTransformOp().Set(gf_matrix(R_S, t_S))
robot_root = add_ref("/World/scene/robot", args.robot)
# mug OUTSIDE the scene, with its grasp pose already written in the USD
mug_root = add_ref("/World/mug", MUG_USD)
if samples:
    _Rp0m, _tp0m = palm_pose(samples[0])
    _Rm0, _tm0 = (C_H2I, np.zeros(3)) if WORLD_VIEW else (_Rp0m.T, -_Rp0m.T @ _tp0m)
    UsdGeom.Xformable(mug_root).AddTransformOp().Set(gf_matrix(_Rm0, _tm0))
simulation_app.update()
_meshes = [str(p.GetPath()) for p in Usd.PrimRange(mug_root) if p.IsA(UsdGeom.Mesh)]
print(f"[isaac_grasp_test] mug meshes: {len(_meshes)} {_meshes[:5]}")

# physics material (the same friction gws_check uses)
mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterial")
pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pm.CreateStaticFrictionAttr(args.mu)
pm.CreateDynamicFrictionAttr(args.mu)
pm.CreateRestitutionAttr(0.0)
for root in (robot_root, mug_root):
    UsdShade.MaterialBindingAPI.Apply(root).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")

# mug: rigid body + collision + mass
UsdPhysics.RigidBodyAPI.Apply(mug_root)
_mug_mass_api = UsdPhysics.MassAPI.Apply(mug_root)
_mug_mass_api.CreateMassAttr(MASS)
if MUG_COM is not None:
    # real centre of mass (mug.json), in the mug frame = Houdini's
    _mug_mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*MUG_COM))
# keep the mug awake: a sleeping body ignores small forces
if "PhysxRigidBodyAPI" not in mug_root.GetAppliedSchemas():
    mug_root.AddAppliedSchema("PhysxRigidBodyAPI")
mug_root.CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
mug_root.CreateAttribute("physxRigidBody:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
mug_meshes = find_prims(mug_root, lambda p: p.IsA(UsdGeom.Mesh))
if not mug_meshes:
    raise RuntimeError("The mug USD has no mesh")
def set_collision_approx(m: Usd.Prim, approx: str):
    """Collision approximation of a mesh, with fine PhysX settings."""
    UsdPhysics.CollisionAPI.Apply(m)
    UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr().Set(approx)
    if approx == "sdf":
        if "PhysxSDFMeshCollisionAPI" not in m.GetAppliedSchemas():
            m.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
        m.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(args.sdf_res)
    elif approx == "convexDecomposition":
        if "PhysxConvexDecompositionCollisionAPI" not in m.GetAppliedSchemas():
            m.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
        m.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(64)
        m.CreateAttribute("physxConvexDecompositionCollision:voxelResolution", Sdf.ValueTypeNames.Int).Set(1000000)
        m.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(True)


for m in mug_meshes:
    set_collision_approx(m, args.mug_collision)
print(f"[isaac_grasp_test] mug collision: {args.mug_collision}")

# gripper: physics variants (USDs from the Isaac 6 importer use them)
def show_and_select_variants(prim: Usd.Prim):
    for p in Usd.PrimRange(prim):
        vsets = p.GetVariantSets()
        for vname in vsets.GetNames():
            vs = vsets.GetVariantSet(vname)
            opts = vs.GetVariantNames()
            sel = vs.GetVariantSelection()
            print(f"[isaac_grasp_test] variant {p.GetPath()} '{vname}': options={opts} selection='{sel}'")
            if "hysic" in vname and sel.lower() in ("", "none"):
                pick = next((o for o in opts if "physx" in o.lower()), None) or next(
                    (o for o in opts if o.lower() != "none"), None)
                if pick:
                    vs.SetVariantSelection(pick)
                    print(f"[isaac_grasp_test]   -> selecting '{pick}'")


show_and_select_variants(robot_root)
simulation_app.update()


def uninstance(root: Usd.Prim):
    """Importer meshes can be instances (not editable): un-instance them.
    Collect the paths first and modify the stage AFTERWARDS (modifying it during
    the traversal invalidates the iterator and crashes Isaac)."""
    paths = [p.GetPath() for p in Usd.PrimRange(root) if p.IsInstance()]
    print(f"[isaac_grasp_test]   instances to un-instance: {len(paths)} {[str(x) for x in paths[:3]]}")
    if not paths:
        return 0
    layer = stage.GetEditTarget().GetLayer()
    with Sdf.ChangeBlock():
        for path in paths:
            spec = Sdf.CreatePrimInLayer(layer, path)
            spec.SetInfo("instanceable", False)
    print("[isaac_grasp_test]   instances removed")
    return len(paths)


_coll = [p for p in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())
         if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
_prev = sorted({str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()) for p in _coll})
print(f"[isaac_grasp_test] finger collision (importer): {len(_coll)} meshes, approximation {_prev}")
if args.finger_collision != "keep":
    n_un = 0
    while True:                                   # nested instances: repeat until none is left
        c = uninstance(robot_root)
        n_un += c
        print(f"[isaac_grasp_test]   instances removed in this pass: {c}")
        if c == 0:
            break
        simulation_app.update()
    _coll = [p for p in Usd.PrimRange(robot_root)
             if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
    print(f"[isaac_grasp_test]   editable collision meshes: {len(_coll)}")
    for p in _coll:
        set_collision_approx(p, args.finger_collision)
    print(f"[isaac_grasp_test] finger collision -> {args.finger_collision} on {len(_coll)} meshes "
          f"({n_un} instances removed)")


def has_art_root(p: Usd.Prim) -> bool:
    return p.HasAPI(UsdPhysics.ArticulationRootAPI) or any("ArticulationRoot" in a for a in p.GetAppliedSchemas())


joints = find_prims(robot_root, lambda p: p.IsA(UsdPhysics.Joint))
bodies = find_prims(robot_root, lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
print(f"[isaac_grasp_test] gripper: {len(joints)} joints, {len(bodies)} rigid bodies")
if not args.keep_inertia:
    _inf = float("-inf")
    for b in bodies:
        ma = UsdPhysics.MassAPI.Apply(b)
        ma.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))      # 0 = computed by PhysX
        ma.CreateCenterOfMassAttr().Set(Gf.Vec3f(_inf, _inf, _inf))     # -inf = computed by PhysX
    print(f"[isaac_grasp_test] inertias and centres of mass: computed by PhysX on {len(bodies)} links "
          "(URDF masses)")
art_roots = find_prims(robot_root, has_art_root)
if not art_roots:
    if not joints or not bodies:
        raise RuntimeError("The gripper has no physics (no joints or rigid bodies): "
                           "is it the USD imported from the URDF? Check the 'variant' lines.")
    UsdPhysics.ArticulationRootAPI.Apply(robot_root)
    art_roots = [robot_root]
    print("[isaac_grasp_test] no ArticulationRootAPI: applied on /World/robot")
art_path = str(art_roots[0].GetPath())
# no self-collisions: convex hulls of neighbouring links (not parent-child)
# overlap near the hinges and push the fingers out of their limits
_ar = art_roots[0]
if "PhysxArticulationAPI" not in _ar.GetAppliedSchemas():
    _ar.AddAppliedSchema("PhysxArticulationAPI")
_ar.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
print("[isaac_grasp_test] gripper self-collisions: disabled")
fixed = find_prims(robot_root, lambda p: p.IsA(UsdPhysics.FixedJoint)
                   and not UsdPhysics.FixedJoint(p).GetBody0Rel().GetTargets())
if not fixed:
    palms = find_prims(robot_root, lambda p: p.GetName() == "palm" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    if not palms:
        raise RuntimeError("Link 'palm' not found to fix the base")
    fj = UsdPhysics.FixedJoint.Define(stage, "/World/fix_palm")
    fj.CreateBody1Rel().SetTargets([palms[0].GetPath()])
    # real palm pose in the world (includes the scene transform
    # and any offset of the link inside the imported USD)
    Wp = UsdGeom.XformCache().GetLocalToWorldTransform(palms[0])
    trp = Gf.Transform(Wp)
    pw = trp.GetTranslation()
    qw = trp.GetRotation().GetQuat()
    sc = trp.GetScale()
    print(f"[isaac_grasp_test] palm in the world: pos=({pw[0]:.4f}, {pw[1]:.4f}, {pw[2]:.4f}) "
          f"escala=({sc[0]:.3f}, {sc[1]:.3f}, {sc[2]:.3f})")
    # palm inside the gripper USD (scene transform removed)
    Sm = UsdGeom.XformCache().GetLocalToWorldTransform(scene_root)
    Lp = Wp * Sm.GetInverse()                    # row vectors: Wp = Lp * Sm
    tl = Lp.ExtractTranslation()
    Rl = np.array([[Lp[i][j] for j in range(3)] for i in range(3)])   # rows = axes
    ang_l = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rl) - 1) / 2))))
    print(f"[isaac_grasp_test] palm INSIDE the gripper USD: pos=({tl[0]:+.4f}, {tl[1]:+.4f}, {tl[2]:+.4f}) "
          f"rotation={ang_l:.1f} deg  (expected: pos 0 and rotation 0)")
    print("[isaac_grasp_test]   palm axes in the USD (rows X, Y, Z):",
          [[round(float(v), 3) for v in r] for r in Rl])
    print(f"[isaac_grasp_test] expected palm (Houdini): pos=({t_S[0]:+.4f}, {t_S[1]:+.4f}, {t_S[2]:+.4f})")
    if max(abs(c - 1.0) for c in sc) > 1e-3:
        print("[isaac_grasp_test] WARNING: the palm scale is != 1; check the gripper USD scale")
    fj.CreateLocalPos0Attr(Gf.Vec3f(float(pw[0]), float(pw[1]), float(pw[2])))
    im = qw.GetImaginary()
    fj.CreateLocalRot0Attr(Gf.Quatf(float(qw.GetReal()), float(im[0]), float(im[1]), float(im[2])))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print("[isaac_grasp_test] base not fixed: added a FixedJoint to the world")
print(f"[isaac_grasp_test] articulation: {art_path}")

# ---------------------------------------------------------------- diagnostics
def h2sim(p_h, Rp, tp):
    """Houdini world point -> simulated world."""
    p_h = np.asarray(p_h, dtype=float)
    return (C_H2I @ p_h) if WORLD_VIEW else (Rp.T @ (p_h - tp))


if args.pose_check and samples:
    s0 = samples[0]
    _Rp0, _tp0 = palm_pose(s0)
    cpos = np.array(s0.get("c_pos", []), dtype=float).reshape(-1, 3)
    cch = s0.get("c_chain", [])
    cgap = s0.get("c_gap", [])
    col = {"thumb": (1.0, 0.0, 0.0), "ring": (0.0, 1.0, 0.0), "index": (0.0, 0.3, 1.0)}
    UsdGeom.Xform.Define(stage, "/World/markers")
    for i, ph in enumerate(cpos):
        ch = cch[i] if i < len(cch) else "?"
        sp = UsdGeom.Sphere.Define(stage, f"/World/markers/c{i}_{ch}")
        sp.CreateRadiusAttr(0.003)
        sp.CreateDisplayColorAttr([Gf.Vec3f(*col.get(ch, (1.0, 1.0, 0.0)))])
        ps = h2sim(ph, _Rp0, _tp0)
        sp.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in ps]))
        g = cgap[i] * 1000 if i < len(cgap) else float("nan")
        print(f"[isaac_grasp_test] contact {i} {ch:6s} zone={s0.get('c_zone', ['?'] * 9)[i]:7s} "
              f"gap_houdini={g:+.2f} mm  pos_sim=({ps[0]:+.4f}, {ps[1]:+.4f}, {ps[2]:+.4f})")
    for nm, ph, cc in (("palma_origen", _tp0, (1.0, 1.0, 1.0)),
                       ("connect", s0["gripper_pose"]["pos"], (1.0, 0.9, 0.0))):
        sp = UsdGeom.Sphere.Define(stage, f"/World/markers/{nm}")
        sp.CreateRadiusAttr(0.005)
        sp.CreateDisplayColorAttr([Gf.Vec3f(*cc)])
        ps = h2sim(ph, _Rp0, _tp0)
        sp.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in ps]))
        print(f"[isaac_grasp_test] marker {nm}: pos_sim=({ps[0]:+.4f}, {ps[1]:+.4f}, {ps[2]:+.4f})")
    # the mug does not move in this mode
    UsdPhysics.RigidBodyAPI(mug_root).CreateKinematicEnabledAttr(True)

# physics: dt and no scene gravity (gravity is applied as a force)
SimulationManager.setup_simulation(dt=args.dt, device="cpu")
for scene in SimulationManager.get_physics_scenes():
    scene.set_gravity((0.0, 0.0, 0.0))

# ---------------------------------------------------------------- look for recording (--look)
def _frame_towards(eye, target, up=(0.0, 0.0, 1.0)):
    """R (columns = local axes) with the local -Z axis looking from eye to target."""
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


LOOK_CAM = "/World/Look/Cam"
if args.gui and args.look and samples:
    # default scene lights: hidden (only the /World/Look ones are seen)
    simulation_app.update()
    _lights = [_p for _p in stage.Traverse()            # collect first, then modify
               if _p.HasAPI(UsdLux.LightAPI) and not str(_p.GetPath()).startswith("/World/Look")]
    for _p in _lights:
        UsdGeom.Imageable(_p).MakeInvisible()
        print(f"[isaac_grasp_test] look: default light hidden {_p.GetPath()}")
    stage.DefinePrim("/World/Look", "Xform")
    g = float(min(max(args.look_bg, 0.0), 1.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/Look/Dome")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(g, g, g * 1.03))
    # framing: between the mug centre and the palm (Isaac world view)
    _s0 = samples[0]
    _mug_c = C_H2I @ np.array(MUG_COM if MUG_COM is not None else _s0.get("obj_center", [0, 0, 0]), float)
    _palm = np.array(t_S, float)
    _aim = 0.65 * _mug_c + 0.35 * _palm
    _span = max(0.12, float(np.linalg.norm(_palm - _mug_c)) + 0.10)
    _hfov = 2.0 * math.atan(18.0 / max(args.look_focal, 1.0))
    _dist = args.look_dist if args.look_dist > 0 else 1.15 * _span / (2.0 * math.tan(_hfov / 2.0)) * 1.6
    _az, _el = math.radians(args.look_az), math.radians(args.look_el)
    _eye = _aim + _dist * np.array([math.cos(_el) * math.cos(_az), math.cos(_el) * math.sin(_az), math.sin(_el)])

    def _light(path, intensity, az_deg, el_deg, angle):
        az, el = math.radians(args.look_az + az_deg), math.radians(el_deg)
        src = _aim + np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        L = UsdLux.DistantLight.Define(stage, path)
        L.CreateIntensityAttr(intensity)
        L.CreateAngleAttr(angle)
        UsdGeom.Xformable(L.GetPrim()).AddTransformOp().Set(gf_matrix(_frame_towards(src, _aim), np.zeros(3)))

    _light("/World/Look/Key", 2600.0, 40.0, 45.0, 3.0)     # key: front-left, high, soft shadow
    _light("/World/Look/Fill", 700.0, -70.0, 15.0, 10.0)   # fill: opposite side, low
    _light("/World/Look/Rim", 1800.0, 180.0, 35.0, 2.0)    # rim: separates the silhouette from the background
    cam = UsdGeom.Camera.Define(stage, LOOK_CAM)
    cam.CreateFocalLengthAttr(float(args.look_focal))
    cam.CreateHorizontalApertureAttr(float(args.look_aperture))
    cam.CreateVerticalApertureAttr(float(args.look_aperture) * 9.0 / 16.0)   # 16:9
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100.0))
    if args.look_cam:
        _v = [float(x) for x in args.look_cam.split(",")]
        if len(_v) != 6:
            raise SystemExit('--look_cam needs 6 numbers: "tx,ty,tz,rx,ry,rz"')
        _xf = UsdGeom.Xformable(cam.GetPrim())
        _xf.AddTranslateOp().Set(Gf.Vec3d(_v[0], _v[1], _v[2]))
        _order = args.look_order.upper()
        _ops = {"XYZ": _xf.AddRotateXYZOp, "XZY": _xf.AddRotateXZYOp, "YXZ": _xf.AddRotateYXZOp,
                "YZX": _xf.AddRotateYZXOp, "ZXY": _xf.AddRotateZXYOp, "ZYX": _xf.AddRotateZYXOp}
        if _order not in _ops:
            raise SystemExit("--look_order must be XYZ, XZY, YXZ, YZX, ZXY or ZYX")
        _ops[_order]().Set(Gf.Vec3f(_v[3], _v[4], _v[5]))
        _dist = float(np.linalg.norm(np.array(_v[:3]) - _aim))
        print(f"[isaac_grasp_test] look: fixed camera --look_cam {args.look_cam}")
    else:
        UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(gf_matrix(_frame_towards(_eye, _aim), _eye))
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = LOOK_CAM
        print(f"[isaac_grasp_test] look: active camera {LOOK_CAM} (dist {_dist:.3f} m, az {args.look_az:g} deg, el {args.look_el:g} deg)")
    except Exception as _e:                                # noqa: BLE001
        print(f"[isaac_grasp_test] look: could not activate the camera ({_e}); pick it in the viewport: {LOOK_CAM}")
    try:
        import carb
        carb.settings.get_settings().set("/app/viewport/grid/enabled", False)
    except Exception:                                      # noqa: BLE001
        pass

# wrappers BEFORE Play: creating them edits the USD (schemas) and, if done
# while simulating, invalidates the physics view
robot = Articulation(art_path)
mug = RigidPrim(str(mug_root.GetPath()))
simulation_app.update()
omni.timeline.get_timeline_interface().play()
simulation_app.update()
simulation_app.update()
if not mug.is_physics_tensor_entity_valid():
    raise RuntimeError("The mug physics view is not valid after Play (check the PhysX warnings above)")
print(f"[isaac_grasp_test] warm-up: {args.warmup} updates (collision cooking)")
for _ in range(args.warmup):
    simulation_app.update()
if args.gui and args.look and args.look_wait > 0:
    print(f"[isaac_grasp_test] look: START RECORDING. The test starts in {args.look_wait:g} s")
    _tw = time.time()
    while time.time() - _tw < args.look_wait and simulation_app.is_running():
        simulation_app.update()
names = list(robot.dof_names)
print("[isaac_grasp_test] DOF:", names)
print(f"[isaac_grasp_test] set-up time (loading + cooking + warm-up): {time.time() - _T0:.1f} s")


def mug_bbox_report(tag):
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = bc.ComputeWorldBound(mug_root).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    sz = mx - mn
    print(f"[isaac_grasp_test] mug {tag}: box min=({mn[0]:+.4f}, {mn[1]:+.4f}, {mn[2]:+.4f}) "
          f"max=({mx[0]:+.4f}, {mx[1]:+.4f}, {mx[2]:+.4f}) size=({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f}) m")
idx = {n: i for i, n in enumerate(names)}
_lims = robot.get_dof_limits()
lo = np.array(_lims[0].numpy()[0], copy=True)   # rad
hi = np.array(_lims[1].numpy()[0], copy=True)
print("[isaac_grasp_test] limits (deg):", {n: (round(math.degrees(lo[i]), 1), round(math.degrees(hi[i]), 1))
                                               for i, n in enumerate(names)})

robot.set_dof_gains(stiffnesses=[args.stiffness] * len(names), dampings=[args.damping] * len(names))
robot.set_dof_max_efforts([args.max_effort] * len(names))

fing = {"thumb": "finger_middle",
        "ring": "finger_2" if args.swap_ring_index else "finger_1",
        "index": "finger_1" if args.swap_ring_index else "finger_2"}
q_open = np.zeros(len(names))
for f in fing.values():
    q_open[idx[f + "_joint_1"]] = 0.0495
    q_open[idx[f + "_joint_2"]] = 0.0
    q_open[idx[f + "_joint_3"]] = -0.0523
dirs_world = {"-Y": (0, -1, 0), "+Y": (0, 1, 0), "+X": (1, 0, 0),
              "-X": (-1, 0, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def step(n: int, force=None, com_local=None):
    for _ in range(n):
        if force is not None:
            p, q = mug.get_world_poses()
            p = p.numpy()[0]
            R = rot_from_wxyz(q.numpy()[0])
            mug.apply_forces_and_torques_at_pos(forces=[force], positions=[R @ com_local + p])
        if args.gui:
            simulation_app.update()
        else:
            SimulationManager.step(steps=1)


def mug_state(com_local):
    p, q = mug.get_world_poses()
    p = p.numpy()[0]
    q = q.numpy()[0]
    R = rot_from_wxyz(q)
    return R @ com_local + p, R


def run_direction(s, Rm, tm, q_target, com_local, dvec, scale=None):
    # mug at its pose relative to the palm, at rest
    cA = Rm @ com_local + tm
    mug.set_world_poses(positions=[tm], orientations=[quat_wxyz(Rm)])
    mug.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
    robot.set_dof_positions(q_open)
    robot.set_dof_velocities(np.zeros(len(names)))
    robot.set_dof_position_targets(q_open)
    step(5)
    robot.set_dof_position_targets(q_target)
    if args.freeze_close:
        _q = [quat_wxyz(Rm)]
        for _ in range(int(args.close_s / args.dt)):
            mug.set_world_poses(positions=[tm], orientations=_q)
            mug.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
            step(1)
        mug.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
        step(int(args.settle_s / args.dt))       # released: settling without force
    else:
        step(int(args.close_s / args.dt))
    c0, R0 = mug_state(com_local)
    close_shift = float(np.linalg.norm(c0 - cA))
    q_closed = np.array(robot.get_dof_positions().numpy()[0], copy=True)
    force = (args.force_scale if scale is None else scale) * MASS * 9.81 * dvec
    step(int(args.hold_s / args.dt), force=force, com_local=com_local)
    c1, R1 = mug_state(com_local)
    disp = float(np.linalg.norm(c1 - c0))
    dR = R1 @ R0.T
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
    q_reached = np.array(robot.get_dof_positions().numpy()[0], copy=True)
    return disp, ang, q_reached, close_shift, q_closed


def print_q(tag, ja=None):
    qn = robot.get_dof_positions().numpy()[0]
    for ours, f in fing.items():
        h = f"Houdini={[round(ja[f'{ours}_0{k}'], 1) for k in (1, 2, 3)]} " if ja else ""
        print(f"[isaac_grasp_test] {tag:14s} {ours:5s} {h}"
              f"Isaac={[round(math.degrees(qn[idx[f'{f}_joint_{k}']]), 1) for k in (1, 2, 3)]}")
    print(f"[isaac_grasp_test] {tag:14s} scissors={[round(math.degrees(qn[idx[n]]), 1) for n in ('palm_finger_1_joint', 'palm_finger_2_joint')]}")


def print_drive(tag):
    st, dp = robot.get_dof_gains()
    me = robot.get_dof_max_efforts().numpy()[0]
    tg = robot.get_dof_position_targets().numpy()[0]
    st, dp = st.numpy()[0], dp.numpy()[0]
    try:
        dt_ = robot.get_dof_drive_types()
    except Exception as e:  # noqa: BLE001
        dt_ = f"(no disponible: {e})"
    for n in names:
        i = idx[n]
        print(f"[isaac_grasp_test] {tag:14s} {n:24s} target={math.degrees(tg[i]):7.1f} deg  "
              f"rigidez={st[i]:8.1f}  amort={dp[i]:6.1f}  par_max={me[i]:6.2f}")
    print(f"[isaac_grasp_test] {tag:14s} drive type: {dt_}")


if args.pose_check:
    s = samples[0]
    print_q("on Play")
    Rp, tp = palm_pose(s)
    Rm, tm = (C_H2I, np.zeros(3)) if WORLD_VIEW else (Rp.T, -Rp.T @ tp)
    ja = s["joint_angles_deg"]
    q_h = q_open.copy()
    for ours, f in fing.items():
        for k in (1, 2, 3):
            q_h[idx[f"{f}_joint_{k}"]] = math.radians(ja[f"{ours}_0{k}"])
    # phase A: mug moved 1 m away -> do the drives hold the angles alone?
    mug.set_world_poses(positions=[tm + np.array([0.0, 0.0, 1.0])], orientations=[quat_wxyz(Rm)])
    robot.set_dof_positions(q_h)
    robot.set_dof_velocities(np.zeros(len(names)))
    robot.set_dof_position_targets(q_h)
    print_q("just set", ja)
    for _ in range(60):
        simulation_app.update()
    print_q("WITHOUT mug", ja)
    print_drive("motor")
    # phase B: mug at its grasp pose
    mug.set_world_poses(positions=[tm], orientations=[quat_wxyz(Rm)])
    robot.set_dof_positions(q_h)
    robot.set_dof_velocities(np.zeros(len(names)))
    robot.set_dof_position_targets(q_h)
    for _ in range(60):
        simulation_app.update()
    mug_bbox_report("en Isaac")
    print(f"[isaac_grasp_test] mug in Houdini: height={s.get('mug_ytop', 0) - s.get('mug_ybot', 0):.4f} m "
          f"(Houdini Y = Isaac Z in the world view)")
    print_q("WITH mug", ja)
    print("[isaac_grasp_test] pose_check: each sphere should touch both its finger and the mug. "
          "Close the window to finish")
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()
    raise SystemExit(0)

FORCE_LEVELS = sorted(float(x) for x in args.force_levels.split(",") if x.strip()) if args.force_levels else []
if FORCE_LEVELS:
    print(f"[isaac_grasp_test] graded test: levels {FORCE_LEVELS} x m*g")
rows_out = []
for s in samples:
    sid = int(s["sample_id"])
    Rp, tp = palm_pose(s)
    if WORLD_VIEW:
        Rm = C_H2I                    # mug in the Isaac world (its Houdini pose, axes converted)
        tm = np.zeros(3)
    else:
        Rm = Rp.T                     # mug in the palm frame
        tm = -Rp.T @ tp
    com_local = np.array(MUG_COM if MUG_COM is not None else s.get("obj_center", [0, 0, 0]), dtype=float)
    ja = s["joint_angles_deg"]
    q_target = q_open.copy()
    for ours, f in fing.items():
        for k in (1, 2, 3):
            v = math.radians(ja[f"{ours}_0{k}"] + args.overclose_deg)
            j = idx[f"{f}_joint_{k}"]
            v = min(max(v, lo[j]), hi[j])
            q_target[j] = v
    res = {"sample_id": sid, "mug_id": s.get("mug_id"), "mass_kg": round(MASS, 5),
           "grasp_label": s["grasp_label"], "grasp_mode": s.get("grasp_mode"),
           "eps_mu05": s.get("eps_mu05"), "eps_mu03": s.get("eps_mu03"),
           "approach_rot_deg": s.get("approach_rot_deg"), "grasp_height_ratio": s.get("grasp_height_ratio")}
    held = 0
    _ts = time.time()
    for dname, dw in dirs_world.items():
        dw = np.array(dw, dtype=float)
        dvec = (C_H2I @ dw) if WORLD_VIEW else (Rp.T @ dw)   # Houdini world direction in the simulated frame
        disp, ang, qr, cshift, qc = run_direction(s, Rm, tm, q_target, com_local, dvec)
        print(f"[isaac_grasp_test]   {dname}: mug moved on closing={cshift * 1000:6.2f} mm  "
              f"during hold={disp * 1000:8.4f} mm  rotation={ang:6.3f} deg")
        n_blk = 0
        for ours, f in fing.items():
            tgt = [math.degrees(q_target[idx[f"{f}_joint_{k}"]]) for k in (1, 2, 3)]
            got = [math.degrees(qc[idx[f"{f}_joint_{k}"]]) for k in (1, 2, 3)]
            blk = any(t - g > 1.0 for t, g in zip(tgt[:2], got[:2]))
            n_blk += int(blk)
            print(f"[isaac_grasp_test]     {ours:5s} target={[round(v, 1) for v in tgt]} "
                  f"reached after closing={[round(v, 1) for v in got]} {'BLOCKED' if blk else 'free'}")
        res[f"blocked_{dname}"] = n_blk
        if n_blk == 0:
            print(f"[isaac_grasp_test]     WARNING: no finger stopped against the mug in {dname}")
        ok = disp < args.max_disp and ang < args.max_rot
        held += int(ok)
        res[f"held_{dname}"] = int(ok)
        res[f"disp_{dname}_mm"] = round(disp * 1000, 4)
        res[f"rot_{dname}_deg"] = round(ang, 2)
        res[f"shift_{dname}_mm"] = round(cshift * 1000, 3)
        if FORCE_LEVELS:
            maxg = args.force_scale if ok else 0.0
            if ok:
                for lvl in FORCE_LEVELS:
                    if lvl <= args.force_scale:
                        continue
                    d2, a2, _, _, _ = run_direction(s, Rm, tm, q_target, com_local, dvec, scale=lvl)
                    if d2 < args.max_disp and a2 < args.max_rot:
                        maxg = lvl
                    else:
                        break
            res[f"maxg_{dname}"] = maxg
            print(f"[isaac_grasp_test]     holds up to {maxg:g} x m*g in {dname}")
    res["held_count"] = held
    if FORCE_LEVELS:
        res["hold_margin_g"] = min(res[f"maxg_{d}"] for d in dirs_world)
    res["min_fingers_blocked"] = min(res[f"blocked_{d}"] for d in dirs_world)
    res["held_gravity"] = res["held_-Y"]
    res["sim_seconds"] = round(time.time() - _ts, 2)
    rows_out.append(res)
    print(f"[isaac_grasp_test] sample {sid:7d} {s['grasp_label']:16s} "
          f"abajo={res['held_-Y']} direcciones={held}/6 desplaz_abajo={res['disp_-Y_mm']} mm"
          + (f" margin={res['hold_margin_g']:g} x m*g" if FORCE_LEVELS else "") + f" t={res['sim_seconds']:.1f}s")
    if args.gui:
        break

# ---------------------------------------------------------------- results
if args.sample >= 0:
    tag = f"sample_{args.sample:04d}"
elif args.samples:
    tag = "list"
elif args.labels:
    tag = args.labels.replace(",", "+")
else:
    tag = "all"
tag += ("_graded" if args.force_levels else "") + \
    (f"_f{args.force_scale:g}" if args.force_scale != 1.0 else "") + (f"_{args.tag}" if args.tag else "") + \
    (f"_mug{MUG_ID:04d}" if MUG_ID is not None else "")
_res_json = args.result_file or os.path.join(args.out, f"isaac_results_{tag}.json")
os.makedirs(os.path.dirname(os.path.abspath(_res_json)), exist_ok=True)
_cfg = dict(vars(args), script_version=SCRIPT_VERSION, mug_usd=MUG_USD, mug_id_used=MUG_ID,
            mass_used=MASS, mass_source=MASS_SRC, com_used=MUG_COM)
_tmp = _res_json + ".tmp"                 # atomic write: never a truncated JSON
with open(_tmp, "w") as fh:
    json.dump({"config": _cfg, "results": rows_out}, fh, indent=1)
os.replace(_tmp, _res_json)
print(f"[isaac_grasp_test] resultados: {_res_json}")
if rows_out:
    with open(os.path.splitext(_res_json)[0] + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
n = len(rows_out)
print(f"[isaac_grasp_test] total time: {time.time() - _T0:.1f} s"
      + (f"  (mean per sample: {sum(r['sim_seconds'] for r in rows_out) / n:.1f} s)" if n else ""))
if n:
    print(f"[isaac_grasp_test] hold with downward gravity: {sum(r['held_gravity'] for r in rows_out)}/{n}")
    print(f"[isaac_grasp_test] hold all 6 directions: {sum(r['held_count'] == 6 for r in rows_out)}/{n}")

if args.gui:
    print("[isaac_grasp_test] --gui mode: close the window to finish")
    while simulation_app.is_running():
        simulation_app.update()
simulation_app.close()
