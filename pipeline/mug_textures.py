# =========================================================================
# mug_textures  v1.1   (external Python with numpy + Pillow; the training venv works)
# Procedural print library for the mug bodies, stored as UDIM tiles.
#
# Each texture is a "full turn" around the mug: u = angle (0..1 around),
# v = height (0 base .. 1 mouth). Saved as UDIM tiles:
#   $HIP/textures/mugtex.1001.png, .1002.png, ...   (tile k -> 1001 + k%10 + 10*(k//10))
# In SOPs each mug gets one tile (uv + offset) and the material reads
#   $HIP/textures/mugtex.<UDIM>.png
# so ONE material serves hundreds of different prints.
#
# Families (all with random palettes): stripes_h, stripes_v, diagonal, dots, plaid,
#   speckle, marble, gradient, dipped, text, shapes, floral, lattice, camo, logo
# Extra output: textures/mugtex_index.json (family and seed of each tile)
#           and textures/mugtex_preview.jpg (contact sheet for review)
# A family that fails does not stop the others (it falls back to stripes).
#
# Usage:
#   python mug_textures.py --out <SYNTH_ROOT>/HIPS/textures --count 200
# Options: --width 1024 --height 512 --seed 0 --fonts <font folder> (default: Windows fonts)
# =========================================================================
import argparse
import colorsys
import faulthandler
import glob
import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

VERSION = "v1.1"
FAMILIES = ["stripes_h", "stripes_v", "diagonal", "dots", "plaid", "speckle", "marble", "gradient",
            "dipped", "text", "shapes", "floral", "lattice", "camo", "logo"]
WORDS = ["CAFE", "COFFEE", "MAMA", "PAPA", "LOVE", "HOME", "TEA", "BEST", "HELLO", "GOOD MORNING", "MONDAY",
         "AMOR", "FELIZ", "HOLA", "BUENOS DIAS", "TE", "ANA", "LUIS", "SOFIA", "2026", "N 1", "OK", "YES",
         "SMILE", "QUITO", "MILK", "CHOCO", "BOSS", "TEAM", "ABC", "XYZ", "WOW", "BB", "M", "A", "Z"]


# ---------------------------------------------------------------- colours
def rand_color(rng, sat=(0.2, 0.9), val=(0.25, 0.95)):
    h = rng.random()
    s = rng.uniform(*sat)
    v = rng.uniform(*val)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def palette(rng, n):
    """Coherent palette: neighbouring or complementary hues, sometimes plus white/black."""
    mode = rng.random()
    base = rng.random()
    cols = []
    for i in range(n):
        if mode < 0.4:
            h = (base + rng.uniform(-0.08, 0.08)) % 1
        elif mode < 0.7:
            h = (base + 0.5 * (i % 2) + rng.uniform(-0.05, 0.05)) % 1
        else:
            h = rng.random()
        s, v = rng.uniform(0.25, 0.95), rng.uniform(0.2, 0.95)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    if rng.random() < 0.35:
        cols[0] = [(245, 245, 240), (235, 230, 220), (20, 20, 22)][int(rng.integers(0, 3))]
    return cols


# ---------------------------------------------------------------- utilities
def value_noise(rng, W, H, cells, octaves=4):
    """Smooth fractal noise in [0,1] (H, W), periodic in u so there is no seam."""
    acc = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cx, cy = max(2, int(cells * 2 ** o)), max(2, int(cells * 2 ** o * H / W) + 1)
        g = rng.random((cy, cx)).astype(np.float32)
        g = np.concatenate([g, g[:, :1]], 1)                    # periodic in u
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((W + W // cx, H), Image.BICUBIC)
        a = np.asarray(im, np.float32)[:, :W] / 255.0
        acc += amp * a
        tot += amp
        amp *= 0.5
    acc /= tot
    return (acc - acc.min()) / max(1e-6, acc.max() - acc.min())


def load_fonts(font_dir):
    names = ["arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "georgia.ttf", "verdana.ttf",
             "verdanab.ttf", "comic.ttf", "comicbd.ttf", "impact.ttf", "segoeui.ttf", "seguisb.ttf",
             "trebuc.ttf", "cour.ttf", "calibri.ttf", "calibrib.ttf"]
    out = []
    for n in names:
        p = os.path.join(font_dir, n)
        if os.path.isfile(p):
            out.append(p)
    if not out:
        out = glob.glob(os.path.join(font_dir, "*.ttf"))[:20]
    return out


def font(fonts, size):
    if fonts:
        try:
            return ImageFont.truetype(random.choice(fonts), size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_paste(dst, src, x, y):
    """Pastes src into dst wrapping in u (for elements that cross the seam)."""
    W = dst.size[0]
    for dx in (-W, 0, W):
        dst.paste(src, (x + dx, y), src)


# ---------------------------------------------------------------- families
def fam_stripes_h(rng, W, H, fonts):
    c = palette(rng, rng.integers(2, 5))
    im = Image.new("RGB", (W, H), c[0])
    d = ImageDraw.Draw(im)
    y = 0
    while y < H:
        h = int(rng.uniform(0.02, 0.18) * H)
        d.rectangle([0, y, W, y + h], fill=c[int(rng.integers(0, len(c)))])
        y += h + int(rng.uniform(0.0, 0.15) * H)
    return im


def fam_stripes_v(rng, W, H, fonts):
    c = palette(rng, rng.integers(2, 4))
    n = int(rng.integers(6, 40))
    im = Image.new("RGB", (W, H), c[0])
    d = ImageDraw.Draw(im)
    w = W / n * rng.uniform(0.2, 0.6)
    for i in range(n):
        x = i * W / n
        d.rectangle([x, 0, x + w, H], fill=c[1 + i % (len(c) - 1)])
    return im


def fam_diagonal(rng, W, H, fonts):
    c = palette(rng, 2)
    im = Image.new("RGB", (W, H), c[0])
    d = ImageDraw.Draw(im)
    n = int(rng.integers(8, 30))
    step = W / n
    w = step * rng.uniform(0.2, 0.5)
    sl = (1 if rng.random() < 0.5 else -1) * rng.uniform(0.3, 1.0) * H
    for i in range(-n, 2 * n):
        x = i * step
        d.polygon([(x, 0), (x + w, 0), (x + w + sl, H), (x + sl, H)], fill=c[1])
    return im


def fam_dots(rng, W, H, fonts):
    c = palette(rng, 3)
    im = Image.new("RGB", (W, H), c[0])
    d = ImageDraw.Draw(im)
    n = int(rng.integers(8, 28))
    sp = W / n
    r = sp * rng.uniform(0.12, 0.38)
    rows = int(H / sp) + 2
    for j in range(rows):
        for i in range(n + 1):
            x = i * sp + (sp / 2 if j % 2 and rng.random() < 0.9 else 0)
            y = j * sp
            col = c[1] if rng.random() < 0.8 else c[2]
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return im


def fam_plaid(rng, W, H, fonts):
    c = palette(rng, 4)
    base = np.zeros((H, W, 3), np.float32) + np.array(c[0], np.float32)
    for k in range(int(rng.integers(3, 8))):
        col = np.array(c[1 + k % 3], np.float32)
        a = rng.uniform(0.25, 0.6)
        if rng.random() < 0.5:
            n = int(rng.integers(4, 14)); w = int(W / n * rng.uniform(0.1, 0.4)); off = int(rng.integers(0, W))
            for i in range(n):
                x0 = (off + int(i * W / n)) % W
                base[:, x0:x0 + w] = base[:, x0:x0 + w] * (1 - a) + col * a
        else:
            n = int(rng.integers(3, 9)); w = int(H / n * rng.uniform(0.1, 0.4)); off = int(rng.integers(0, H))
            for i in range(n):
                y0 = (off + int(i * H / n)) % H
                base[y0:y0 + w] = base[y0:y0 + w] * (1 - a) + col * a
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def fam_speckle(rng, W, H, fonts):
    c = palette(rng, 3)
    a = np.zeros((H, W, 3), np.float32) + np.array(c[0], np.float32)
    n = int(rng.uniform(0.0005, 0.006) * W * H)
    xs, ys = rng.integers(0, W, n), rng.integers(0, H, n)
    m = np.zeros((H, W), np.float32)
    m[ys, xs] = 1
    r = [0, 1, 1, 2][int(rng.integers(0, 4))]               # dilation in numpy (no MaxFilter: it killed the process on Windows)
    if r:
        acc = m.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                acc = np.maximum(acc, np.roll(np.roll(m, dy, 0), dx, 1))
        m = acc
    col = np.array(c[1] if rng.random() < 0.7 else (30, 25, 20), np.float32)
    a = a * (1 - m[..., None]) + col * m[..., None]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def fam_marble(rng, W, H, fonts):
    c = palette(rng, 2)
    n = value_noise(rng, W, H, rng.uniform(2, 5), 5)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = rng.uniform(4, 14)
    t = np.sin((xx / W * rng.uniform(-1, 1) + yy / H) * f + n * rng.uniform(4, 10))
    v = np.abs(t) ** rng.uniform(0.3, 1.5)
    a = np.array(c[0], np.float32) * v[..., None] + np.array(c[1], np.float32) * (1 - v[..., None])
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def fam_gradient(rng, W, H, fonts):
    c = palette(rng, 2)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None] ** rng.uniform(0.5, 2.0)
    t = np.repeat(t, W, 1)
    a = np.array(c[0], np.float32) * (1 - t[..., None]) + np.array(c[1], np.float32) * t[..., None]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def fam_dipped(rng, W, H, fonts):
    c = palette(rng, 2)
    xs = np.arange(W, dtype=np.float32)
    edge = rng.uniform(0.2, 0.7) * H + np.sin(xs / W * 2 * math.pi * rng.integers(1, 6)
                                              + rng.uniform(0, 6)) * rng.uniform(0, 0.08) * H
    edge += (value_noise(rng, W, 1, 6, 3)[0] - 0.5) * rng.uniform(0, 0.1) * H
    yy = np.arange(H, dtype=np.float32)[:, None]
    soft = rng.uniform(1, 12)
    m = np.clip((edge[None, :] - (H - yy)) / soft + 0.5, 0, 1)        # bottom (small v) = colour 1
    a = np.array(c[0], np.float32) * (1 - m[..., None]) + np.array(c[1], np.float32) * m[..., None]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def fam_text(rng, W, H, fonts):
    c = palette(rng, 3)
    im = Image.new("RGB", (W, H), c[0])
    for _ in range(int(rng.integers(1, 14))):
        s = int(rng.uniform(0.06, 0.45) * H)
        f = font(fonts, s)
        txt = random.choice(WORDS)
        lay = Image.new("RGBA", (int(s * len(txt) * 0.8) + 10, int(s * 1.4)), (0, 0, 0, 0))
        ImageDraw.Draw(lay).text((5, 0), txt, font=f, fill=c[1 + int(rng.integers(0, 2))] + (255,))
        if rng.random() < 0.3:
            lay = lay.rotate(rng.uniform(-30, 30), expand=True)
        wrap_paste(im, lay, int(rng.integers(0, W)), int(rng.integers(0, max(1, H - lay.size[1]))))
    return im


def shape(d, kind, x, y, r, col, rng):
    if kind == "circle":
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    elif kind == "square":
        d.rectangle([x - r, y - r, x + r, y + r], fill=col)
    elif kind == "triangle":
        d.polygon([(x, y - r), (x + r, y + r), (x - r, y + r)], fill=col)
    elif kind == "star":
        pts = []
        for k in range(10):
            a = math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((x + rr * math.cos(a), y - rr * math.sin(a)))
        d.polygon(pts, fill=col)
    elif kind == "heart":
        d.ellipse([x - r, y - r * 0.8, x, y], fill=col)
        d.ellipse([x, y - r * 0.8, x + r, y], fill=col)
        d.polygon([(x - r * 0.98, y - r * 0.3), (x + r * 0.98, y - r * 0.3), (x, y + r)], fill=col)
    elif kind == "leaf":
        d.ellipse([x - r * 0.4, y - r, x + r * 0.4, y + r], fill=col)


def fam_shapes(rng, W, H, fonts):
    c = palette(rng, 4)
    im = Image.new("RGB", (W, H), c[0])
    d = ImageDraw.Draw(im)
    kinds = ["circle", "square", "triangle", "star", "heart", "leaf"]
    use = [kinds[i] for i in rng.choice(len(kinds), int(rng.integers(1, 4)), replace=False)]
    for _ in range(int(rng.integers(5, 60))):
        r = rng.uniform(0.03, 0.15) * H
        shape(d, random.choice(use), rng.uniform(0, W), rng.uniform(0, H), r, c[1 + int(rng.integers(0, 3))], rng)
    return im


def fam_floral(rng, W, H, fonts):
    c = palette(rng, 5)
    im = Image.new("RGB", (W, H), c[0])
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(4, 30))):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.04, 0.14) * H
        npet = int(rng.integers(4, 9))
        pc = c[1 + int(rng.integers(0, 3))]
        for k in range(npet):
            a = 2 * math.pi * k / npet
            px, py = x + r * math.cos(a), y + r * math.sin(a)
            d.ellipse([px - r * 0.6, py - r * 0.6, px + r * 0.6, py + r * 0.6], fill=pc)
        d.ellipse([x - r * 0.45, y - r * 0.45, x + r * 0.45, y + r * 0.45], fill=c[4])
        if rng.random() < 0.6:
            shape(d, "leaf", x + r * 1.6, y + r * 0.8, r * 0.9, c[int(rng.integers(1, 4))], rng)
    return im


def fam_lattice(rng, W, H, fonts):
    c = palette(rng, 2)
    if rng.random() < 0.6:                                   # same hue, lighter/darker (embossed look)
        r, g, b = c[0]
        k = rng.uniform(0.7, 1.3)
        c[1] = tuple(int(min(255, max(0, v * k))) for v in (r, g, b))
    im = Image.new("RGB", (W, H), c[0])
    d = ImageDraw.Draw(im)
    n = int(rng.integers(10, 40))
    step = W / n
    w = max(1, int(step * rng.uniform(0.06, 0.25)))
    for i in range(-n, 2 * n):
        x = i * step
        d.line([(x, 0), (x + H, H)], fill=c[1], width=w)
        d.line([(x, 0), (x - H, H)], fill=c[1], width=w)
    return im


def fam_camo(rng, W, H, fonts):
    c = palette(rng, 4)
    n = value_noise(rng, W, H, rng.uniform(3, 8), 3)
    q = np.quantile(n, [0.3, 0.55, 0.8])
    idx = np.digitize(n, q)
    a = np.array(c, np.float32)[idx]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def fam_logo(rng, W, H, fonts):
    c = palette(rng, 3)
    im = Image.new("RGB", (W, H), c[0])
    sides = [0.25, 0.75] if rng.random() < 0.6 else [rng.uniform(0, 1)]
    for u in sides:
        if rng.random() < 0.5:
            s = int(rng.uniform(0.2, 0.55) * H)
            f = font(fonts, s)
            txt = random.choice(WORDS)
            lay = Image.new("RGBA", (int(s * len(txt) * 0.8) + 10, int(s * 1.4)), (0, 0, 0, 0))
            ImageDraw.Draw(lay).text((5, 0), txt, font=f, fill=c[1] + (255,))
        else:
            r = int(rng.uniform(0.15, 0.35) * H)
            lay = Image.new("RGBA", (2 * r + 4, 2 * r + 4), (0, 0, 0, 0))
            shape(ImageDraw.Draw(lay), random.choice(["circle", "star", "heart", "square"]), r + 2, r + 2, r,
                  c[1 + int(rng.integers(0, 2))] + (255,), rng)
        x = int(u * W - lay.size[0] / 2)
        y = int(rng.uniform(0.25, 0.75) * H - lay.size[1] / 2)
        wrap_paste(im, lay, x, y)
    return im


FUNCS = {n: globals()["fam_" + n] for n in FAMILIES}


def udim(k):
    return 1001 + (k % 10) + 10 * (k // 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fonts", default=r"C:\Windows\Fonts")
    a = ap.parse_args()
    faulthandler.enable()                                   # if something crashes in C, report where
    os.makedirs(a.out, exist_ok=True)
    fonts = load_fonts(a.fonts)
    print("fonts found: %d" % len(fonts))
    index = []
    thumbs = []
    for k in range(a.count):
        fam = FAMILIES[k % len(FAMILIES)]
        s = a.seed * 100003 + k * 7919 + 17
        rng = np.random.default_rng(s)
        random.seed(s)
        try:
            im = FUNCS[fam](rng, a.width, a.height, fonts).convert("RGB")
        except Exception as e:                              # a broken family does not stop the others
            print("[warning] tile %d (%s) failed: %r -> using stripes" % (k, fam, e))
            im = fam_stripes_h(rng, a.width, a.height, fonts).convert("RGB")
            fam = "stripes_h"
        if rng.random() < 0.3:                                  # some wear / non-uniformity
            n = value_noise(rng, a.width, a.height, 6, 3)
            arr = np.asarray(im, np.float32) * (1 + (n[..., None] - 0.5) * rng.uniform(0.05, 0.2))
            im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        u = udim(k)
        im.save(os.path.join(a.out, "mugtex.%d.png" % u))
        index.append({"tile": k, "udim": u, "family": fam, "seed": int(s)})
        thumbs.append(im.resize((256, 128)))
        if (k + 1) % 20 == 0:
            print("  %d / %d" % (k + 1, a.count), flush=True)
    with open(os.path.join(a.out, "mugtex_index.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": VERSION, "count": a.count, "size": [a.width, a.height],
                   "udim_rule": "1001 + k%10 + 10*(k//10)", "tiles": index}, fh, indent=1)
    cols = 10
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 256, rows * 128), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 128))
    sheet.save(os.path.join(a.out, "mugtex_preview.jpg"), quality=85)
    print("%d textures -> %s   (check mugtex_preview.jpg)" % (a.count, a.out))


if __name__ == "__main__":
    main()
