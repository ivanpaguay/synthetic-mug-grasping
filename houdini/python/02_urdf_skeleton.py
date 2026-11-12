# =========================================================================
# urdf_skeleton  v1.2   (Python SOP, no inputs)
# Houdini node: /obj/geo5/python1
#
# Reads the Robotiq 3F URDF and builds the KineFX skeleton with the SAME names
# as the rig (wrist, connect, thumb/ring/index _01.._04), plus one _00 point per
# finger (finger base, link_0) outside the chains. s@urdf_link says which URDF
# mesh is bound rigidly to each point. The kinematics come from the file, as
# in Isaac Sim: nothing is interpreted by hand.
#
# Parameters (Edit Parameter Interface):
#   urdf_path       (string)  path to the .urdf
#   swap_ring_index (toggle)  0: finger_1 = ring, finger_2 = index; 1: swapped
#   scissor_deg     (float)   scissor (palm_finger_1 = +s, palm_finger_2 = -s)
#   open_pose       (toggle)  0 (default): URDF zero pose, every joint at 0 =
#                             the pose in which Blender and Isaac place the
#                             meshes -> use it as REST for the binding.
#                             1: real open pose (joint_1 at its lower limit
#                             2.8 deg, joint_3 at its upper limit -3 deg).
#
# Units: metres. Axes: the URDF palm axes (fingers grow towards +Y, as in Houdini).
# =========================================================================
import math
import xml.etree.ElementTree as ET

try:
    import hou
except ImportError:          # lets the script be tested outside Houdini
    hou = None

# distal tip in the link_3 frame, measured on the official collision mesh
# (m). The distal link has no joint at its tip.
TIP_LOCAL = (0.0295, 0.0231, 0.0)
# palm: front face (connect) and mounting base (wrist), on the central axis
CONNECT_LOCAL = (0.0, 0.0508, 0.0)
WRIST_LOCAL = (0.0, -0.0540, 0.0)


# ---------- minimal algebra (4x4 matrices as lists, column vectors) ----------
def mat_rpy_xyz(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    R = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]]
    return [R[0] + [xyz[0]], R[1] + [xyz[1]], R[2] + [xyz[2]], [0, 0, 0, 1]]


def mat_axis_angle(axis, q):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s, t = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0],
            [0, 0, 0, 1]]


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def apply(M, p):
    return tuple(M[i][0] * p[0] + M[i][1] * p[1] + M[i][2] * p[2] + M[i][3] for i in range(3))


def rot_axis(M, a):
    return tuple(M[i][0] * a[0] + M[i][1] * a[1] + M[i][2] * a[2] for i in range(3))


def floats(s, n=3):
    v = [float(t) for t in (s or "").split()]
    return tuple(v) if len(v) == n else (0.0,) * n


# ---------- reading the URDF ----------
def read_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        l = j.find("limit")
        joints[j.get("name")] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": floats(o.get("xyz")) if o is not None else (0.0, 0.0, 0.0),
            "rpy": floats(o.get("rpy")) if o is not None else (0.0, 0.0, 0.0),
            "axis": floats(a.get("xyz")) if a is not None else (1.0, 0.0, 0.0),
            "lo": float(l.get("lower")) if l is not None and l.get("lower") else 0.0,
            "hi": float(l.get("upper")) if l is not None and l.get("upper") else 0.0,
        }
    return joints


def forward(joints, q):
    """World frame of each link and frame of each joint (at q = 0 of that joint)."""
    by_child = {j["child"]: (n, j) for n, j in joints.items()}
    link_T, joint_T = {}, {}

    def link_frame(link):
        if link in link_T:
            return link_T[link]
        if link not in by_child:                      # root (palm)
            link_T[link] = mat_rpy_xyz((0, 0, 0), (0, 0, 0))
            return link_T[link]
        name, j = by_child[link]
        Tj = mul(link_frame(j["parent"]), mat_rpy_xyz(j["xyz"], j["rpy"]))
        joint_T[name] = Tj
        qj = q.get(name, 0.0) if j["type"] in ("revolute", "continuous") else 0.0
        link_T[link] = mul(Tj, mat_axis_angle(j["axis"], qj))
        return link_T[link]

    for j in joints.values():
        link_frame(j["child"])
    return link_T, joint_T


def build(path, swap=False, scissor_deg=0.0, open_pose=False):
    J = read_urdf(path)
    s = math.radians(scissor_deg)
    q = {}
    for n, j in J.items():
        if not open_pose:
            break
        if n.endswith("joint_1"):
            q[n] = j["lo"]
        elif n.endswith("joint_3"):
            q[n] = j["hi"]
    if "palm_finger_1_joint" in J:
        q["palm_finger_1_joint"] = max(J["palm_finger_1_joint"]["lo"], min(J["palm_finger_1_joint"]["hi"], s))
    if "palm_finger_2_joint" in J:
        q["palm_finger_2_joint"] = max(J["palm_finger_2_joint"]["lo"], min(J["palm_finger_2_joint"]["hi"], -s))
    LT, JT = forward(J, q)

    fingers = [("thumb", "finger_middle", "palm_finger_middle_joint"),
               ("index" if swap else "ring", "finger_1", "palm_finger_1_joint"),
               ("ring" if swap else "index", "finger_2", "palm_finger_2_joint")]
    pts = []   # (name, chain, P, R (3x3, rows = axes), extras)
    palm = LT["palm"]
    pts.append(("wrist", "", apply(palm, WRIST_LOCAL), palm, {"urdf_link": "palm"}))
    pts.append(("connect", "", apply(palm, CONNECT_LOCAL), palm, {"urdf_link": ""}))
    for ours, f, base_joint in fingers:
        chain = "chain_" + ours
        bj = J[base_joint]
        bT = JT[base_joint]
        base = {"urdf_base_joint": base_joint,
                "scissor_type": bj["type"],
                "scissor_pivot": apply(bT, (0, 0, 0)),
                "scissor_axis": rot_axis(LT[f + "_link_0"], bj["axis"]),
                "scissor_lo": bj["lo"], "scissor_hi": bj["hi"],
                "scissor_q": q.get(base_joint, 0.0)}
        # _00: finger base (link_0). Turns with the scissor; fixed on the thumb.
        # NO chain attribute: the chains keep 4 points (_01.._04).
        L0 = LT[f + "_link_0"]
        ex0 = dict(base)
        ex0["urdf_link"] = f + "_link_0"
        pts.append(("%s_00" % ours, "", apply(L0, (0, 0, 0)), L0, ex0))
        for k in (1, 2, 3):
            jn = "%s_joint_%d" % (f, k)
            j = J[jn]
            L = LT["%s_link_%d" % (f, k)]
            ex = {"urdf_joint": jn, "joint_lo": j["lo"], "joint_hi": j["hi"],
                  "joint_q": q.get(jn, 0.0), "joint_axis": rot_axis(L, j["axis"]),
                  "urdf_link": "%s_link_%d" % (f, k)}
            if k == 1:
                ex.update(base)
            pts.append(("%s_%02d" % (ours, k), chain, apply(L, (0, 0, 0)), L, ex))
        L3 = LT[f + "_link_3"]
        pts.append(("%s_04" % ours, chain, apply(L3, TIP_LOCAL), L3, {"urdf_joint": "", "urdf_link": ""}))
    return pts


def rows(M):
    """Local axes as rows (Houdini 3@transform convention)."""
    return tuple(M[i][c] for c in range(3) for i in range(3))


# ---------- Houdini ----------
if hou is not None:
    node = hou.pwd()
    geo = node.geometry()
    path = node.evalParm("urdf_path") if node.parm("urdf_path") else ""
    swap = bool(node.evalParm("swap_ring_index")) if node.parm("swap_ring_index") else False
    scd = node.evalParm("scissor_deg") if node.parm("scissor_deg") else 0.0
    opn = bool(node.evalParm("open_pose")) if node.parm("open_pose") else False
    if not path:
        raise hou.NodeError("urdf_path is missing")
    pts = build(hou.text.expandString(path), swap, scd, opn)

    geo.addAttrib(hou.attribType.Point, "name", "")
    geo.addAttrib(hou.attribType.Point, "chain", "")
    ta = geo.addAttrib(hou.attribType.Point, "transform", (1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    try:
        ta.setOption("type", "matrix")
    except Exception:
        pass
    for a, d in (("urdf_joint", ""), ("urdf_link", ""), ("urdf_base_joint", ""), ("scissor_type", "")):
        geo.addAttrib(hou.attribType.Point, a, d)
    for a in ("joint_lo", "joint_hi", "joint_q", "scissor_lo", "scissor_hi", "scissor_q"):
        geo.addAttrib(hou.attribType.Point, a, 0.0)
    for a in ("joint_axis", "scissor_axis", "scissor_pivot"):
        geo.addAttrib(hou.attribType.Point, a, (0.0, 0.0, 0.0))

    index = {}
    for nm, chain, P, M, ex in pts:
        p = geo.createPoint()
        p.setPosition(P)
        p.setAttribValue("name", nm)
        p.setAttribValue("chain", chain)
        p.setAttribValue("transform", rows(M))
        for k, v in ex.items():
            p.setAttribValue(k, v)
        index[nm] = p

    def bone(a, b):
        poly = geo.createPolygon(False)
        poly.addVertex(index[a])
        poly.addVertex(index[b])

    bone("wrist", "connect")
    for f in ("thumb", "ring", "index"):
        bone("wrist", f + "_00")
        bone(f + "_00", f + "_01")
        for k in (1, 2, 3):
            bone("%s_%02d" % (f, k), "%s_%02d" % (f, k + 1))

    geo.addAttrib(hou.attribType.Global, "urdf_path", path)
    geo.addAttrib(hou.attribType.Global, "gripper_id", "robotiq_3f_urdf")
    geo.addAttrib(hou.attribType.Global, "skeleton_version", "urdf_skeleton v1.2")
    geo.addAttrib(hou.attribType.Global, "skeleton_pose", "open" if opn else "zero")
