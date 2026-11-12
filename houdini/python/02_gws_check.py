# =========================================================================
# gws_check  v4.1   (Python SOP)   input: grasp_contacts
# Houdini node: /obj/geo5/gws_check1
#
# Question: does the grasp RESIST? (force closure)
# Method: Ferrari-Canny epsilon in the grasp wrench space
#   - each contact pushes inside its friction cone (8 edges)
#   - each edge gives a 6-value wrench: force (3) + torque (3)
#   - torque normalised by the largest contact radius
#   - epsilon = distance from the origin to the boundary of the wrench hull
#       epsilon > 0  -> force closure (resists in every direction)
#       epsilon <= 0 -> there is a direction in which the object escapes
# Reference thresholds:
#   Dex-Net 2.0: mu = 0.5, epsilon >= 0.002 -> positive grasp
#   robustness (this project): epsilon >= 0.002 also with mu = 0.3
# The label it writes is GEOMETRIC; the final label comes from the physics
# test in Isaac Sim (isaac/grasp_physics_test.py).
# =========================================================================
import hou
import math
import numpy as np

MU      = 0.5     # nominal friction (Dex-Net 2.0)
MU_LOW  = 0.3     # low friction: smooth / wet surface
EDGES   = 8       # friction cone edges
EPS_MIN = 0.002   # quality threshold (Dex-Net 2.0)
TORSION = 0.0025  # soft finger: torsional friction = mu x patch radius (0.5 x 5 mm)

geo = hou.pwd().geometry()


def gval(name, default):
    return geo.attribValue(name) if geo.findGlobalAttrib(name) is not None else default


def setg(name, value):
    if geo.findGlobalAttrib(name) is None:
        if isinstance(value, str):
            geo.addAttrib(hou.attribType.Global, name, "")
        elif isinstance(value, int):
            geo.addAttrib(hou.attribType.Global, name, 0)
        else:
            geo.addAttrib(hou.attribType.Global, name, 0.0)
    geo.setGlobalAttribValue(name, value)


def wrench_set(P, N, c, mu, torsion=0.0):
    """Wrenches of the friction-cone edges. N points INTO the object.
    torsion > 0 adds the soft-finger model (the pad resists spinning about its normal)."""
    rho = max(float(np.linalg.norm(P - c, axis=1).max()), 1e-6)
    W = []
    for p, n in zip(P, N):
        n = n / max(np.linalg.norm(n), 1e-12)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        r = p - c
        for k in range(EDGES):
            ang = 2.0 * math.pi * k / EDGES
            f = n + mu * (math.cos(ang) * t1 + math.sin(ang) * t2)
            f /= np.linalg.norm(f)
            W.append(np.concatenate([f, np.cross(r, f) / rho]))
        if torsion > 0.0:
            for sgn in (1.0, -1.0):
                W.append(np.concatenate([n, (np.cross(r, n) + sgn * torsion * n) / rho]))
    return np.array(W)


try:
    from scipy.spatial import ConvexHull
    BACKEND = "scipy"
except Exception:
    ConvexHull = None
    BACKEND = "sampling"

_U = None


def epsilon(W):
    """Distance from the origin to the convex hull boundary (<= 0: no force closure)."""
    global _U
    if len(W) < 7:
        return -1.0
    if ConvexHull is not None:
        try:
            h = ConvexHull(W, qhull_options="QJ")
            return float((-h.equations[:, -1]).min())
        except Exception:
            return -1.0
    # without scipy: support function sampled in 6D (approximation)
    if _U is None:
        rng = np.random.default_rng(0)
        _U = rng.normal(size=(20000, 6))
        _U /= np.linalg.norm(_U, axis=1, keepdims=True)
    return float((_U @ W.T).max(axis=1).min())


# ---------- data from grasp_contacts ----------
P = np.array(gval("c_pos", ()), dtype=float).reshape(-1, 3)
N = np.array(gval("c_nrm", ()), dtype=float).reshape(-1, 3)
c = np.array(gval("obj_center", (0.0, 0.0, 0.0)), dtype=float)

start = int(gval("start_collision", 0))
pen   = float(gval("pen_depth", 0.0))
selfc = int(gval("self_collision", 0))
table = int(gval("table_hit", 0))
opp   = int(gval("opposition", 0))
nc    = len(P)

# ---------- contact geometry diagnostics ----------
# deviation from a line: if the contacts are nearly collinear, the object can
# spin about that line and point contacts cannot stop it
line_dev = 0.0
spread = 0.0
if nc >= 2:
    Q = P - P.mean(axis=0)
    sv = np.linalg.svd(Q, compute_uv=False)
    line_dev = float(sv[1] / math.sqrt(nc)) if len(sv) > 1 else 0.0
    spread = float(max(np.linalg.norm(P[i] - P[j]) for i in range(nc) for j in range(i + 1, nc)))

# ---------- label (priority from top to bottom) ----------
eps, eps_low, eps_soft = -1.0, -1.0, -1.0
if start:
    label, stage = "invalid_start_collision", "generator"
elif pen > 0:
    label, stage = "invalid_penetration", "geometry"
elif selfc:
    label, stage = "invalid_self_collision", "geometry"
elif table:
    label, stage = "invalid_table_collision", "geometry"
elif nc < 3 or not opp:
    label, stage = "invalid_missing_contact", "contact"
else:
    eps = epsilon(wrench_set(P, N, c, MU))
    eps_soft = epsilon(wrench_set(P, N, c, MU, TORSION))   # informative, does not decide the label
    if eps < EPS_MIN:
        label, stage = "invalid_no_force_closure", "physics"
    else:
        eps_low = epsilon(wrench_set(P, N, c, MU_LOW))
        label = "valid_robust" if eps_low >= EPS_MIN else "valid"
        stage = "ok"

# ---------- approach path: separate flag ----------
path_col = int(gval("path_collision", 0))
path_tab = int(gval("path_table_hit", 0))
approach_clear = 0 if (path_col or path_tab) else 1
setg("approach_clear", approach_clear)
setg("sample_valid", 1 if (label.startswith("valid") and approach_clear) else 0)

setg("grasp_label", label)
setg("fail_stage", stage)
setg("grasp_valid", 1 if label.startswith("valid") else 0)
setg("eps_mu05", eps)
setg("eps_mu03", eps_low)
setg("eps_soft", eps_soft)
setg("contact_line_dev", line_dev)
setg("contact_spread", spread)
setg("gws_torsion", TORSION)
setg("gws_mu", MU)
setg("gws_mu_low", MU_LOW)
setg("gws_eps_min", EPS_MIN)
setg("gws_backend", BACKEND)
setg("gws_version", "v4.1")
