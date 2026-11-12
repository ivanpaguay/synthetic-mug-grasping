# =========================================================================
# profile_style  v1.1   (Python Script TOP, between wedge1 and make_r_top)
# Houdini node: /obj/geo5/topnet1/profile_style      input: mug_id_offset
# Mug types. It takes the values wedge1 randomizes and, depending on the type,
# REMAPS them into that type's ranges (same work item = always the same mug).
# This way profile_curve and handle_generator keep reading @glass_* and never
# disagree.
#
#   profile_style 0 = regular mug       (55 %)  -> the wedge ranges, unchanged
#                 1 = tea cup           (15 %)  low, very open mouth, foot
#                 2 = espresso / low    (15 %)  small, almost straight, sometimes a foot
#                 3 = bowl-like         (15 %)  wide, bulging
# v1.1: EXACT proportions (a deck): with N mugs, 55 % regular and the rest
#       split evenly between tea, espresso and bowl; the deck is shuffled with
#       a fixed seed and each mug takes the card at its position (mug_num).
# Writes to the work item (overwrites): glass_h, glass_r_base, glass_top_ratio,
#   glass_flare, glass_bulge; and new: profile_style, glass_foot_h, glass_foot_r
# (glass_r_top is still computed by make_r_top = r_base * top_ratio)
# Evaluate Script During: Cook (In-Process)
# =========================================================================
import random

wi = work_item


def g(name, default=0.0):
    v = wi.attribValue(name)
    return float(v) if v is not None else default


def remap(v, a0, a1, b0, b1):
    t = (v - a0) / (a1 - a0) if a1 != a0 else 0.5
    t = min(1.0, max(0.0, t))
    return b0 + t * (b1 - b0)


idx = wi.attribValue("mug_num")
idx = int(idx) if idx is not None else int(wi.index)
tot = wi.attribValue("mug_total")
tot = int(tot) if tot is not None else 40
rng = random.Random(1000003 + idx * 7919)          # for the foot and per-mug details

# deck of types with exact proportions
n_other = int(round(0.45 * tot))
n1 = n_other // 3
n2 = n_other // 3
n3 = n_other - n1 - n2
deck = [1] * n1 + [2] * n2 + [3] * n3
deck = [0] * (tot - len(deck)) + deck
random.Random(424242).shuffle(deck)

# wedge1 ranges (to remap without losing the randomized variation)
H0, H1 = 0.088, 0.13
RB0, RB1 = 0.030, 0.045
TR0, TR1 = 0.9, 1.3
FL0, FL1 = 0.6, 2.5
BU0, BU1 = -0.004, 0.008

h, rb, tr, fl, bu = g("glass_h"), g("glass_r_base"), g("glass_top_ratio"), g("glass_flare"), g("glass_bulge")

style = deck[idx % len(deck)]
foot_h, foot_r = 0.0, 0.0

if style == 1:      # tea cup: low, very open mouth, foot
    h = remap(h, H0, H1, 0.055, 0.075)
    rb = remap(rb, RB0, RB1, 0.020, 0.026)
    tr = remap(tr, TR0, TR1, 1.60, 2.10)
    fl = remap(fl, FL0, FL1, 0.35, 0.70)        # opens quickly -> curved profile
    bu = remap(bu, BU0, BU1, 0.000, 0.003)
    foot_h = rng.uniform(0.005, 0.009)
    foot_r = rb * rng.uniform(1.00, 1.15)
elif style == 2:    # espresso / low mug
    h = remap(h, H0, H1, 0.055, 0.070)
    rb = remap(rb, RB0, RB1, 0.027, 0.033)
    tr = remap(tr, TR0, TR1, 1.00, 1.20)
    fl = remap(fl, FL0, FL1, 0.80, 1.60)
    bu = remap(bu, BU0, BU1, -0.001, 0.003)
    if rng.random() < 0.5:
        foot_h = rng.uniform(0.003, 0.006)
        foot_r = rb * rng.uniform(0.85, 0.95)
elif style == 3:    # bowl-like
    h = remap(h, H0, H1, 0.070, 0.090)
    rb = remap(rb, RB0, RB1, 0.034, 0.044)
    tr = remap(tr, TR0, TR1, 1.15, 1.40)
    fl = remap(fl, FL0, FL1, 0.70, 1.30)
    bu = remap(bu, BU0, BU1, 0.002, 0.008)

wi.setIntAttrib("profile_style", style)
wi.setFloatAttrib("glass_h", h)
wi.setFloatAttrib("glass_r_base", rb)
wi.setFloatAttrib("glass_top_ratio", tr)
wi.setFloatAttrib("glass_flare", fl)
wi.setFloatAttrib("glass_bulge", bu)
wi.setFloatAttrib("glass_foot_h", foot_h)
wi.setFloatAttrib("glass_foot_r", foot_r)
