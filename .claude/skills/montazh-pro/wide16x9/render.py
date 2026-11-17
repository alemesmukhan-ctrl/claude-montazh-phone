# -*- coding: utf-8 -*-
"""Абай жолы — 16:9 динамичный монтаж «Кино-эпик».
usage:  python render.py sketch t1 t2 ...   -> sketch frames
        python render.py render OUT.mp4      -> full render (4 workers)
"""
import os, sys, json, math, subprocess, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(D + "/../..")
FONTS = ROOT + "/.claude/skills/montazh-pro/fonts/"
GRADED = D + "/graded.mp4"
W, H, FPS = 1920, 1080, 30.0
DUR = 50.245
NFR = int(round(DUR * FPS))
GOLD = (230, 180, 60)
GOLD_L = (255, 226, 140)
GOLD_D = (150, 98, 24)
TEAL_BG = (8, 34, 30)

# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0): return a if x < a else b if x > b else x
def ease(x): x = clamp(x); return x * x * (3 - 2 * x)
def eout(x): x = clamp(x); return 1 - (1 - x) ** 3
def ein(x): x = clamp(x); return x ** 3
def eback(x, s=1.7):
    x = clamp(x); x -= 1; return x * x * ((s + 1) * x + s) + 1
def lerp(a, b, p): return a + (b - a) * p
def prog(t, t0, d): return clamp((t - t0) / d)
def window(t, s, e, fi=0.25, fo=0.25):
    """0..1 visibility envelope"""
    return min(eout(prog(t, s, fi)), 1 - ein(prog(t, e - fo, fo)))

# ------------------------------------------------------------------ fonts / text
@functools.lru_cache(None)
def font(name, size):
    if name.startswith("/"):
        return ImageFont.truetype(name, size)
    return ImageFont.truetype(FONTS + "Montserrat-%s.ttf" % name, size)

def to_np(img):
    return np.asarray(img, dtype=np.float32) / 255.0

def text_w(txt, f, track=0):
    if track == 0:
        return f.getlength(txt)
    return sum(f.getlength(c) for c in txt) + track * (len(txt) - 1)

@functools.lru_cache(None)
def text_rgba(txt, size, fname="ExtraBold", fill=(255, 255, 255), grad=None, track=0,
              extrude=0, ext_col=(90, 55, 10), shadow=0.55, glow=None, stroke=0, stroke_col=(0, 0, 0)):
    """returns float32 RGBA (straight alpha) and anchor offset (ox, oy) of text origin"""
    f = font(fname, size)
    asc, desc = f.getmetrics()
    tw = int(math.ceil(text_w(txt, f, track)))
    pad = int(size * 0.5) + extrude * 2 + 10
    cw, ch = tw + 2 * pad, asc + desc + 2 * pad
    m = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(m)
    if track == 0:
        d.text((pad, pad), txt, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    else:
        x = pad
        for c in txt:
            d.text((x, pad), c, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
            x += f.getlength(c) + track
    M = to_np(m)
    out = np.zeros((ch, cw, 4), np.float32)

    def put(rgb, a):
        a = a[..., None]
        col = np.asarray(rgb, np.float32)[:3] / 255
        na = out[..., 3:] * (1 - a) + a
        out[..., :3] = (out[..., :3] * out[..., 3:] * (1 - a) + col * a) / np.maximum(1e-6, na)
        out[..., 3:] = na

    if shadow:
        r = max(2, int(size * 0.14))
        s = cv2.GaussianBlur(M, (0, 0), r)
        s = np.roll(s, int(size * 0.07) + extrude, axis=0)
        put((0, 0, 0), s * shadow)
    if glow:
        g = cv2.GaussianBlur(M, (0, 0), size * 0.25) * glow[1]
        put(glow[0], np.clip(g, 0, 1))
    for i in range(extrude, 0, -1):
        sh = np.roll(np.roll(M, i, axis=0), int(i * 0.5), axis=1)
        k = 0.55 + 0.45 * (1 - i / max(1, extrude))
        put(tuple(int(c * k) for c in ext_col), sh)
    if grad is not None:
        top, bot = np.array(grad[0], np.float32) / 255, np.array(grad[1], np.float32) / 255
        ys = np.linspace(0, 1, ch, dtype=np.float32)
        y0, y1 = pad / ch, (pad + asc) / ch
        p = np.clip((ys - y0) / max(1e-3, y1 - y0), 0, 1)[:, None, None]
        col = top * (1 - p) + bot * p
        a = M[..., None]
        out[..., :3] = (out[..., :3] * out[..., 3:] * (1 - a) + col * a) / np.maximum(1e-6, out[..., 3:] * (1 - a) + a)
        out[..., 3:] = out[..., 3:] * (1 - a) + a
    else:
        put(fill, M)
    return out, (pad, pad), (tw, asc + desc)

# ------------------------------------------------------------------ compositing
def over(dst, rgba, x, y, op=1.0):
    """straight-alpha rgba patch onto float RGB dst at integer x,y"""
    h, w = rgba.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0 or op <= 0.002:
        return
    p = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = p[..., 3:4] * op
    reg = dst[y0:y1, x0:x1]
    reg *= (1 - a)
    reg += p[..., :3] * a

def over_pm(dst, pm, x, y, op=1.0):
    """premultiplied rgba"""
    h, w = pm.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0 or op <= 0.002:
        return
    p = pm[y0 - y:y1 - y, x0 - x:x1 - x] * op
    reg = dst[y0:y1, x0:x1]
    reg *= (1 - p[..., 3:4])
    reg += p[..., :3]

def premul(rgba):
    o = rgba.copy(); o[..., :3] *= o[..., 3:4]; return o

def warp(dst, rgba, cx, cy, rx=0.0, ry=0.0, rz=0.0, s=1.0, op=1.0, f=1700.0, pm=None):
    """draw patch with 3D rotation (degrees) centred at cx,cy"""
    if op <= 0.002 or s <= 0.01:
        return
    h, w = rgba.shape[:2]
    if pm is None:
        pm = premul(rgba)
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    pts = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], np.float64) * s
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    P = pts @ (Ry @ Rx @ Rz).T
    z = P[:, 2] + f
    if np.any(z < 50):
        return
    dx = P[:, 0] * f / z + cx
    dy = P[:, 1] * f / z + cy
    bx0, by0 = int(math.floor(dx.min())) - 2, int(math.floor(dy.min())) - 2
    bx1, by1 = int(math.ceil(dx.max())) + 2, int(math.ceil(dy.max())) + 2
    cx0, cy0 = max(0, bx0), max(0, by0)
    cx1, cy1 = min(dst.shape[1], bx1), min(dst.shape[0], by1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dd = np.float32(np.stack([dx - cx0, dy - cy0], 1))
    Mx = cv2.getPerspectiveTransform(src, dd)
    out = cv2.warpPerspective(pm, Mx, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    over_pm(dst, out, cx0, cy0, op)

def text_draw(dst, txt, cx, cy, size, op=1.0, anchor="c", rx=0, ry=0, rz=0, s=1.0, **kw):
    img, (ox, oy), (tw, th) = text_rgba(txt, size, **kw)
    h, w = img.shape[:2]
    # centre of text box inside patch
    tcx, tcy = ox + tw / 2, oy + th / 2
    if anchor == "l":
        cx = cx + tw / 2 * s
    elif anchor == "r":
        cx = cx - tw / 2 * s
    if rx == 0 and ry == 0 and rz == 0 and abs(s - 1) < 1e-3:
        over(dst, img, cx - tcx, cy - tcy, op)
    else:
        # shift so text centre is the pivot
        pcx = cx + (w / 2 - tcx) * s
        pcy = cy + (h / 2 - tcy) * s
        warp(dst, img, pcx, pcy, rx, ry, rz, s, op)
    return tw * s, th * s

# ------------------------------------------------------------------ vector drawing helpers (PIL, 3x supersample)
SS = 3
def vec_canvas(w, h):
    im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)
def vec_done(im):
    w, h = im.width // SS, im.height // SS
    return to_np(im.resize((w, h), Image.LANCZOS))
def S_(*v): return [x * SS for x in v]

def rounded_mask(w, h, r):
    m = np.zeros((h * 2, w * 2), np.uint8)
    r2 = int(r * 2)
    if r2 * 2 >= min(w, h) * 2:
        cv2.ellipse(m, (w, h), (w, h), 0, 0, 360, 255, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(m, (r2, 0), (w * 2 - r2, h * 2), 255, -1)
        cv2.rectangle(m, (0, r2), (w * 2, h * 2 - r2), 255, -1)
        for cx, cy in ((r2, r2), (w * 2 - r2 - 1, r2), (r2, h * 2 - r2 - 1), (w * 2 - r2 - 1, h * 2 - r2 - 1)):
            cv2.circle(m, (cx, cy), r2, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

# ------------------------------------------------------------------ assets
@functools.lru_cache(None)
def asset(name, height=None, width=None):
    im = Image.open(D + "/assets/" + name).convert("RGBA")
    if height:
        im = im.resize((int(im.width * height / im.height), height), Image.LANCZOS)
    elif width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    return to_np(im)

@functools.lru_cache(None)
def framed(name, height, border=6, radius=10, glow=True):
    """photo with gold border, rounded corners and drop-shadow padding"""
    a = asset(name, height=height).copy()
    h, w = a.shape[:2]
    m = rounded_mask(w, h, radius)
    a[..., 3] *= m
    pad = 40
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    sh = np.zeros(out.shape[:2], np.float32)
    sh[pad:pad + h, pad:pad + w] = m
    sh = cv2.GaussianBlur(sh, (0, 0), 16)
    out[..., 3] = sh * 0.7
    if glow:
        out[..., 0] = 0.9 * sh; out[..., 1] = 0.65 * sh; out[..., 2] = 0.2 * sh
        out[..., :3] /= np.maximum(sh[..., None], 1e-4)
        out[..., 3] = sh * 0.55
    # composite photo
    reg = out[pad:pad + h, pad:pad + w]
    al = a[..., 3:4]
    rgb = reg[..., :3] * reg[..., 3:4] * (1 - al) + a[..., :3] * al
    na = reg[..., 3:4] * (1 - al) + al
    reg[..., :3] = rgb / np.maximum(na, 1e-6); reg[..., 3:4] = na
    # gold border
    bm = np.zeros((h, w), np.float32)
    inner = np.zeros((h, w), np.float32)
    inner[border:h - border, border:w - border] = 1
    edge = np.clip(m - cv2.erode(m, np.ones((border, border), np.uint8)), 0, 1) * 0.9
    col = np.array(GOLD_L, np.float32) / 255
    e = edge[..., None]
    reg[..., :3] = reg[..., :3] * (1 - e) + col * e
    return out

# ------------------------------------------------------------------ words & chunks
CHUNKS = [
    "Қазақ әдебиетінде", "өз орнын айрықша", "қалыптастырған шығармалардың бірі —", "Мұхтар Әуезовтің",
    "*«Абай жолы»* роман-эпопеясы",
    "Бүгін біз осы ұлы шығарманың", "*бірінші томына* тоқталамыз",
    "Романның алғашқы томында", "Абайдың балалық шағы,", "өскен ортасы,", "отбасылық тәрбиесі", "және өмірге деген көзқарасының",
    "қалыптасуы баяндалады.",
    "Шығармада *Құнанбайдың* болмысы,", "Зере мен Ұлжанның тәрбиесі,", "қазақ қоғамындағы", "әлеуметтік қатынастар мен",
    "сол кезеңнің салт-дәстүрлері", "кеңінен көрініс табады.",
    "Кітаптың басты ерекшелігі —", "бір адамның өмір жолы арқылы", "*тұтас бір дәуірдің*", "тыныс-тіршілігін танытуында.",
    "Айта кетейік,", "осы кітапқа арналған", "анимациялық жарнамалық бейнеролик", "*Claude Code* көмегімен әзірленді.",
    "Әдеби мұраны", "жаңа технологиямен таныту —", "*өткен мен бүгінді*", "байланыстырудың бір жолы.",
    "Бұл — менің", "кітапқа арналған жарнамам.",
]
EXTRA = [("бұл", 45.85, 46.1), ("менің", 46.1, 46.5), ("кітапқа", 47.2, 47.65), ("арналған", 47.65, 48.1), ("жарнамам", 48.15, 49.0)]

def norm(s):
    return "".join(c for c in s.lower() if c.isalpha())

def build_chunks():
    wj = json.load(open(D + "/words.json"))["words"]
    ws = [(norm(w["w"]), w["s"], w["e"]) for w in wj if norm(w["w"])]
    ws += [(norm(a), s, e) for a, s, e in EXTRA]
    i = 0
    chunks = []
    for c in CHUNKS:
        items = []
        acc = False
        for tok in c.split(" "):
            a0 = tok.startswith("*")
            if a0: acc = True
            disp = tok.replace("*", "")
            n = norm(disp)
            if not n:  # dash etc. glue to previous
                if items: items[-1]["w"] += " " + disp
                continue
            while i < len(ws) and ws[i][0] != n:
                i += 1
            assert i < len(ws), ("no match", disp)
            items.append(dict(w=disp, s=ws[i][1], e=ws[i][2], a=acc))
            if tok.endswith("*") or tok.endswith("*,") or "*" in tok[1:]:
                acc = False
            i += 1
        chunks.append(dict(items=items, s=items[0]["s"], e=items[-1]["e"]))
    for k, c in enumerate(chunks):
        nxt = chunks[k + 1]["s"] if k + 1 < len(chunks) else DUR
        c["show"] = max(0.0, c["s"] - 0.08)
        c["hide"] = min(nxt - 0.02, max(c["e"] + 0.45, c["show"] + 1.0))
        if k + 1 == len(chunks):
            c["hide"] = min(DUR, c["e"] + 0.6)
    return chunks

CH = None

# ------------------------------------------------------------------ face track
FACES = None
def load_faces():
    global FACES
    f = json.load(open(D + "/faces.json"))
    k = H / f["fh"]
    S = f["samples"]
    t = np.array([s["t"] for s in S])
    arr = {key: np.array([s[key] * k for s in S]) for key in ["x0", "x1", "y0", "y1", "top"]}
    # smooth (moving avg 3)
    for key in arr:
        a = arr[key]; b = a.copy()
        b[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3
        arr[key] = b
    FACES = (t, arr)

def face_at(t):
    ts, arr = FACES
    r = {k: float(np.interp(t, ts, v)) for k, v in arr.items()}
    r["cx"] = (r["x0"] + r["x1"]) / 2
    r["cy"] = (r["y0"] + r["y1"]) / 2
    return r

def face_slow(t):
    """heavily smoothed face centre (camera pivot)"""
    xs = [face_at(t + d)["cx"] for d in np.linspace(-0.8, 0.8, 9)]
    ys = [face_at(t + d)["cy"] for d in np.linspace(-0.8, 0.8, 9)]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# ------------------------------------------------------------------ timeline
# modes: full / pip / bg
SEGS = [
    (0.00, 4.70, "full", None),
    (4.70, 7.05, "bg", None),
    (7.05, 10.05, "pip", dict(rect=(110, 120, 820, 580), r=36)),
    (10.05, 13.95, "full", None),
    (13.95, 17.95, "pip", dict(rect=(1140, 100, 600, 600), r=300)),
    (17.95, 22.05, "bg", None),
    (22.05, 27.05, "full", None),
    (27.05, 31.70, "pip", dict(rect=(130, 120, 560, 560), r=30)),
    (31.70, 34.30, "full", None),
    (34.30, 36.95, "bg", None),
    (36.95, 39.25, "full", None),
    (39.25, 42.65, "pip", dict(rect=(1010, 120, 800, 570), r=36)),
    (42.65, 49.00, "full", None),
    (49.00, DUR + 1, "bg", None),
]
TR = 0.45  # pip transition

def seg_at(t):
    for i, s in enumerate(SEGS):
        if s[0] <= t < s[1]:
            return i, s
    return len(SEGS) - 1, SEGS[-1]

# camera zoom keys for full mode: (t, zoom)
ZK = [(0.0, 1.0), (3.6, 1.07), (4.7, 1.07),
      (10.05, 1.13), (11.0, 1.08), (11.8, 1.0), (13.95, 1.0),
      (22.05, 1.0), (24.9, 1.04), (25.15, 1.12), (27.05, 1.14),
      (31.70, 1.0), (32.05, 1.0), (32.25, 1.12), (33.15, 1.12), (33.35, 1.0), (34.3, 1.04),
      (36.95, 1.0), (39.25, 1.0),
      (42.65, 1.0), (43.2, 1.0), (43.45, 1.1), (45.8, 1.1), (46.0, 1.0), (49.0, 1.09)]
HARD = {10.05, 25.15, 32.25, 43.45}

def zoom_at(t):
    for (t0, z0), (t1, z1) in zip(ZK, ZK[1:]):
        if t0 <= t <= t1:
            if t1 in HARD or (t1 - t0) < 0.3:
                return lerp(z0, z1, eout(prog(t, t0, t1 - t0)))
            return lerp(z0, z1, ease(prog(t, t0, t1 - t0)))
    return ZK[-1][1]

def cam(t):
    z = zoom_at(t)
    px, py = face_slow(t)
    return z, px, py

def cam_apply(img, z, px, py):
    if abs(z - 1) < 1e-4:
        return img
    M = np.float32([[z, 0, px - px * z], [0, z, py - py * z]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def cam_pt(x, y, z, px, py):
    return px + (x - px) * z, py + (y - py) * z

# ------------------------------------------------------------------ static layers
@functools.lru_cache(None)
def bottom_grad():
    y = np.linspace(0, 1, H, dtype=np.float32)
    a = np.clip((y - 0.45) / 0.55, 0, 1)
    a = a * a * (3 - 2 * a) * 0.80
    return a[:, None, None]

@functools.lru_cache(None)
def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.70)) ** 2)
    return (1 - 0.38 * np.clip(d - 0.55, 0, 1) ** 1.3)[..., None]

@functools.lru_cache(None)
def grain(i):
    rng = np.random.default_rng(i)
    n = rng.standard_normal((H // 2, W // 2)).astype(np.float32) * 0.018
    return cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]

@functools.lru_cache(None)
def bg_base():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    top = np.array([12, 46, 40], np.float32) / 255
    bot = np.array([3, 12, 11], np.float32) / 255
    p = (yy / H)[..., None]
    g = top * (1 - p) + bot * p
    d = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.42) / H) ** 2)[..., None]
    g = g + np.array([0.10, 0.08, 0.03], np.float32) * np.clip(0.55 - d, 0, 1) * 1.4
    return g.astype(np.float32)

@functools.lru_cache(None)
def ornament():
    R = 520
    im, d = vec_canvas(2 * R, 2 * R)
    c = R * SS
    col = (240, 200, 110, 255)
    for rr, wdt in ((500, 3), (470, 1), (330, 2), (300, 1)):
        d.ellipse([c - rr * SS, c - rr * SS, c + rr * SS, c + rr * SS], outline=col, width=wdt * SS)
    # ram-horn curls between rings
    for k in range(16):
        ang = 2 * math.pi * k / 16
        for sgn in (1, -1):
            pts = []
            for i in range(60):
                u = i / 59
                th = sgn * u * 4.2
                rad = 70 * (1 - u * 0.8)
                lx = 400 + rad * math.cos(th) - 70
                ly = rad * math.sin(th)
                # rotate into ring position
                gx = c + SS * (lx * math.cos(ang) - ly * math.sin(ang))
                gy = c + SS * (lx * math.sin(ang) + ly * math.cos(ang))
                pts.append((gx, gy))
            d.line(pts, fill=col, width=3 * SS, joint="curve")
        a2 = ang + math.pi / 16
        rx, ry = c + SS * 485 * math.cos(a2), c + SS * 485 * math.sin(a2)
        d.ellipse([rx - 6 * SS, ry - 6 * SS, rx + 6 * SS, ry + 6 * SS], fill=col)
    return vec_done(im)

@functools.lru_cache(None)
def particles():
    rng = np.random.default_rng(3)
    n = 170
    return dict(x=rng.uniform(0, W, n), y=rng.uniform(0, H, n), z=rng.uniform(0.3, 1.0, n),
                ph=rng.uniform(0, 6.28, n), sp=rng.uniform(10, 40, n))

def draw_bg(t, ornament_on=True, orn_xy=(960, 430), orn_op=0.13):
    g = bg_base().copy()
    # moving warm light
    lx = W * (0.5 + 0.25 * math.sin(t * 0.35))
    ly = H * 0.3
    # particles (half-res layer)
    P = particles()
    lay = np.zeros((H // 2, W // 2), np.float32)
    ys = (P["y"] - t * P["sp"] * P["z"] * 1.6) % H
    xs = (P["x"] + 25 * np.sin(t * 0.6 + P["ph"]) * P["z"]) % W
    tw = 0.5 + 0.5 * np.sin(t * 3 + P["ph"] * 3)
    for x, y, z, b in zip(xs, ys, P["z"], tw):
        cv2.circle(lay, (int(x / 2), int(y / 2)), max(1, int(z * 2.2)), float(0.35 + 0.65 * b) * z, -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.6) * 1.8 + cv2.GaussianBlur(lay, (0, 0), 6) * 2.5
    lay = cv2.resize(lay, (W, H))[..., None]
    g += lay * np.array([1.0, 0.78, 0.35], np.float32) * 0.55
    if ornament_on:
        o = ornament()
        ang = t * 8
        M = cv2.getRotationMatrix2D((o.shape[1] / 2, o.shape[0] / 2), ang, 1.0)
        ro = cv2.warpAffine(premul(o), M, (o.shape[1], o.shape[0]))
        over_pm(g, ro, int(orn_xy[0] - o.shape[1] / 2), int(orn_xy[1] - o.shape[0] / 2), orn_op)
    return g

# ------------------------------------------------------------------ person mask (behind text)
_SEG = None
def person_alpha(rgb_u8):
    global _SEG
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    if _SEG is None:
        _SEG = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=ROOT + "/.claude/skills/montazh-pro/models/selfie_multiclass.tflite"),
            output_confidence_masks=True))
    r = _SEG.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8)))
    a = 1.0 - r.confidence_masks[0].numpy_view().astype(np.float32)
    if a.ndim == 3: a = a[..., 0]
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.clip((a - 0.3) / 0.4, 0, 1)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    return a[..., None]

BEHIND = [  # (s, e, text, kind)
    (11.85, 13.95, "БАЛАЛЫҚ", "rise"),
    (36.95, 39.25, "CLAUDE CODE", "type"),
]

def draw_behind(dst, t, b, face, z, px, py):
    s, e, txt, kind = b
    vis = window(t, s, e, 0.2, 0.3)
    if vis <= 0:
        return
    top = cam_pt(face["cx"], face["top"], z, px, py)
    fcx = cam_pt(face["cx"], face["cy"], z, px, py)[0]
    size = 250 if kind == "rise" else 210
    f = font("Black", size)
    asc = f.getmetrics()[0]
    # glyph visual bottom ~ baseline; head covers the lower ~45%
    cy = top[1] + size * 0.05
    cy = max(cy, size * 0.42 + 18)
    # darken background behind for legibility
    band = np.zeros((H, 1, 1), np.float32)
    yy = np.arange(H, dtype=np.float32)
    band[:, 0, 0] = np.exp(-((yy - cy) / (size * 0.9)) ** 2) * 0.45 * vis
    dst *= (1 - band)
    if kind == "rise":
        n = len(txt)
        tw_total = text_w(txt, f, 6)
        x = W / 2 - tw_total / 2
        x = clamp(fcx - tw_total / 2, 40, W - 40 - tw_total)
        for i, c in enumerate(txt):
            p = prog(t, s + i * 0.045, 0.42)
            cw = f.getlength(c)
            op = eout(p) * (1 - ein(prog(t, e - 0.3, 0.3)))
            text_draw(dst, c, x + cw / 2, cy + (1 - eback(p)) * 120, size, op=op, rx=(1 - eout(p)) * 75,
                      fname="Black", grad=((255, 241, 196), GOLD), extrude=8, ext_col=(110, 70, 15), shadow=0.6)
            x += cw + 6
    else:
        n = int(clamp((t - s) / 0.06, 0, len(txt)))
        shown = txt[:n]
        full_w = text_w(txt, f, 4)
        x0 = clamp(fcx - full_w / 2, 40, W - 40 - full_w)
        if shown.strip():
            text_draw(dst, shown, x0, cy, size, op=vis, anchor="l", fname="Black", track=4,
                      grad=((255, 255, 255), (255, 214, 150)), extrude=6, ext_col=(120, 70, 30), shadow=0.6)
        # cursor
        cw_ = text_w(shown, f, 4) if shown else 0
        if (n < len(txt) or int(t * 3) % 2 == 0) and vis > 0.5:
            cv2.rectangle(dst, (int(x0 + cw_ + 10), int(cy - size * 0.36)), (int(x0 + cw_ + 26), int(cy + size * 0.36)),
                          (1.0, 0.85, 0.5), -1)

# ------------------------------------------------------------------ subtitles
def chunk_layout(c, size=60, asize=72):
    items = []
    x = 0
    for it in c["items"]:
        sz = asize if it["a"] else size
        fn = "Black" if it["a"] else "ExtraBold"
        txt = it["w"].upper() if it["a"] else it["w"]
        w_ = font(fn, sz).getlength(txt)
        items.append(dict(it, txt=txt, sz=sz, fn=fn, x=x, wd=w_))
        x += w_ + 26
    total = x - 26
    for i in items:
        i["x"] -= total / 2
    return items, total

def draw_subs(dst, t, ycenter):
    for c in CH:
        if not (c["show"] <= t < c["hide"]):
            continue
        items, total = chunk_layout(c)
        out_p = prog(t, c["hide"] - 0.12, 0.12)
        for it in items:
            p = prog(t, it["s"] - 0.06, 0.24)
            if p <= 0:
                continue
            op = eout(p) * (1 - out_p)
            sc = 0.7 + 0.3 * eback(p, 1.4)
            dy = (1 - eout(p)) * 26 - out_p * 14
            cx = W / 2 + it["x"] + it["wd"] / 2
            if it["a"]:
                text_draw(dst, it["txt"], cx, ycenter + dy, it["sz"], op=op, s=sc, rx=(1 - eout(p)) * 70,
                          fname=it["fn"], grad=(GOLD_L, (214, 150, 40)), extrude=5, ext_col=(100, 62, 12), shadow=0.75,
                          glow=((255, 190, 80), 0.35))
            else:
                text_draw(dst, it["txt"], cx, ycenter + dy, it["sz"], op=op, s=sc,
                          fname=it["fn"], fill=(255, 255, 255), shadow=0.8)

# ------------------------------------------------------------------ small graphics
def label_tag(dst, t, s, e, txt, sub, x, y, side="l", anchor_pt=None):
    """face-tracked label with connector line; side 'l' = label left of point"""
    vis = window(t, s, e, 0.3, 0.25)
    if vis <= 0:
        return
    f = font("ExtraBold", 40)
    tw = text_w(txt, f, 3)
    wipe = eout(prog(t, s, 0.45))
    bx = x - tw - 30 if side == "l" else x + 30
    # connector
    if anchor_pt is not None:
        ax, ay = anchor_pt
        lx = x
        p2 = (int(lerp(ax, lx, wipe)), int(lerp(ay, y, wipe)))
        cv2.line(dst, (int(ax), int(ay)), p2, (1.0, 0.82, 0.45), 3, cv2.LINE_AA)
        cv2.circle(dst, (int(ax), int(ay)), 8, (1.0, 0.85, 0.5), -1, cv2.LINE_AA)
        cv2.circle(dst, (int(ax), int(ay)), int(8 + 18 * (0.5 + 0.5 * math.sin(t * 6))), (1.0, 0.85, 0.5), 2, cv2.LINE_AA)
    # glass backing
    pw, ph = int(tw + 50), 70 + (40 if sub else 0)
    gx = int(bx - 25 if side == "l" else bx - 25)
    gy = int(y - 35)
    sub_rect = dst[max(0, gy):gy + ph, max(0, gx):gx + int(pw * wipe)]
    if sub_rect.size:
        sub_rect *= (1 - 0.55 * vis)
        sub_rect += np.array([0.02, 0.06, 0.05], np.float32) * vis
        cv2.rectangle(dst, (gx, gy), (gx + 6, gy + ph), (0.9, 0.7, 0.25), -1)
    if wipe > 0.3:
        text_draw(dst, txt, bx, y, 40, op=vis * eout(prog(t, s + 0.12, 0.3)), anchor="l", fname="ExtraBold", track=3,
                  fill=(255, 236, 190), shadow=0.6)
        if sub:
            text_draw(dst, sub, bx, y + 44, 28, op=vis * eout(prog(t, s + 0.2, 0.3)) * 0.85, anchor="l",
                      fname="Medium", fill=(230, 230, 230), shadow=0.5)

def gold_burst(dst, t, t0, cx, cy, r0=60, r1=360):
    p = prog(t, t0, 0.6)
    if p <= 0 or p >= 1:
        return
    r = int(lerp(r0, r1, eout(p)))
    a = (1 - p) * 0.9
    lay = np.zeros((H, W), np.float32)
    cv2.circle(lay, (int(cx), int(cy)), r, a, int(3 + 6 * (1 - p)), cv2.LINE_AA)
    for k in range(12):
        ang = k * math.pi / 6 + 0.3
        r2a, r2b = r * 0.75, r * 1.05
        cv2.line(lay, (int(cx + r2a * math.cos(ang)), int(cy + r2a * math.sin(ang))),
                 (int(cx + r2b * math.cos(ang)), int(cy + r2b * math.sin(ang))), a, 3, cv2.LINE_AA)
    lay = lay[..., None]
    dst += lay * np.array([1.0, 0.8, 0.4], np.float32)

def flash(dst, t, times):
    for T in times:
        d = abs(t - T)
        if d < 0.16:
            k = (1 - d / 0.16) ** 2 * 0.75
            dst += k * np.array([1.0, 0.9, 0.7], np.float32)

# ---------- icons (vector)
@functools.lru_cache(None)
def icon(name, size=110, col=(245, 205, 110)):
    im, d = vec_canvas(size, size)
    c = col + (255,)
    s = SS
    L = lambda *p: [v * s for v in p]
    w = 5 * s
    if name == "child":
        d.ellipse(L(40, 14, 70, 44), outline=c, width=w)
        d.line(L(55, 44, 55, 76), fill=c, width=w)
        d.line(L(32, 56, 78, 56), fill=c, width=w)
        d.line(L(55, 76, 40, 98), fill=c, width=w); d.line(L(55, 76, 70, 98), fill=c, width=w)
        d.polygon(L(84, 20, 88, 30, 98, 31, 90, 37, 93, 47, 84, 41, 75, 47, 78, 37, 70, 31, 80, 30), fill=c)
    elif name == "yurt":
        d.arc(L(12, 18, 98, 80), 180, 360, fill=c, width=w)
        d.line(L(12, 49, 12, 92), fill=c, width=w); d.line(L(98, 49, 98, 92), fill=c, width=w)
        d.line(L(8, 92, 102, 92), fill=c, width=w)
        d.rectangle(L(44, 60, 66, 92), outline=c, width=w)
        for xx in range(20, 95, 12):
            d.line(L(xx, 54, xx + 8, 88), fill=c, width=2 * s)
        d.ellipse(L(47, 14, 63, 24), outline=c, width=3 * s)
    elif name == "family":
        for (x, y, r) in ((30, 30, 13), (80, 30, 13), (55, 55, 10)):
            d.ellipse(L(x - r, y - r, x + r, y + r), outline=c, width=w)
        d.arc(L(8, 48, 52, 100), 200, 340, fill=c, width=w)
        d.arc(L(58, 48, 102, 100), 200, 340, fill=c, width=w)
        d.arc(L(38, 68, 72, 104), 200, 340, fill=c, width=w)
    elif name == "eye":
        d.arc(L(8, 30, 102, 110), 205, 335, fill=c, width=w)
        d.arc(L(8, 0, 102, 80), 25, 155, fill=c, width=w)
        d.ellipse(L(40, 40, 70, 70), outline=c, width=w)
        d.ellipse(L(50, 50, 60, 60), fill=c)
    elif name == "person":
        d.ellipse(L(38, 10, 72, 44), fill=c)
        d.pieslice(L(18, 52, 92, 130), 180, 360, fill=c)
    elif name == "book":
        d.polygon(L(55, 30, 10, 18, 10, 88, 55, 100), outline=c, width=w)
        d.polygon(L(55, 30, 100, 18, 100, 88, 55, 100), outline=c, width=w)
        for yy in (40, 54, 68):
            d.line(L(18, yy - 8, 46, yy), fill=c, width=3 * s); d.line(L(64, yy, 92, yy - 8), fill=c, width=3 * s)
    elif name == "chip":
        d.rounded_rectangle(L(26, 26, 84, 84), radius=8 * s, outline=c, width=w)
        d.rounded_rectangle(L(40, 40, 70, 70), radius=4 * s, fill=c)
        for k in range(4):
            v = 34 + k * 14
            d.line(L(v, 8, v, 26), fill=c, width=3 * s); d.line(L(v, 84, v, 102), fill=c, width=3 * s)
            d.line(L(8, v, 26, v), fill=c, width=3 * s); d.line(L(84, v, 102, v), fill=c, width=3 * s)
    elif name == "network":
        pts = [(20, 30), (85, 22), (55, 60), (20, 92), (92, 88)]
        for i, j in ((0, 2), (1, 2), (2, 3), (2, 4), (0, 1), (3, 4)):
            d.line(L(*pts[i], *pts[j]), fill=c, width=3 * s)
        for (x, y) in pts:
            d.ellipse(L(x - 9, y - 9, x + 9, y + 9), fill=c)
    return vec_done(im.resize((size * SS, size * SS)) if False else im)

@functools.lru_cache(None)
def icon_badge(name, size=104):
    """icon inside circular dark-glass badge with gold ring"""
    pad = 20
    S2 = size + 2 * pad
    im, d = vec_canvas(S2, S2)
    s = SS
    d.ellipse([pad * s, pad * s, (pad + size) * s, (pad + size) * s], fill=(10, 30, 26, 230), outline=(240, 200, 110, 255), width=4 * s)
    base = vec_done(im)
    ic = icon(name, int(size * 0.66))
    o = int(pad + size * 0.17)
    over_rgba(base, ic, o, o)
    return base

def over_rgba(dst, src, x, y):
    h, w = src.shape[:2]
    reg = dst[y:y + h, x:x + w]
    a = src[..., 3:4]
    na = reg[..., 3:4] * (1 - a) + a
    reg[..., :3] = (reg[..., :3] * reg[..., 3:4] * (1 - a) + src[..., :3] * a) / np.maximum(na, 1e-6)
    reg[..., 3:4] = na

# ---------- character cards
@functools.lru_cache(None)
def char_card(kind):
    cw, chh = 360, 480
    im, d = vec_canvas(cw, chh)
    s = SS
    L = lambda *p: [v * s for v in p]
    d.rounded_rectangle(L(0, 0, cw, chh), radius=22 * s, fill=(12, 40, 34, 245), outline=(240, 200, 110, 255), width=4 * s)
    d.rounded_rectangle(L(14, 14, cw - 14, chh - 14), radius=14 * s, outline=(240, 200, 110, 110), width=2 * s)
    # corner curls
    for (x, y, a0) in ((34, 34, 180), (cw - 34, 34, 270), (34, chh - 34, 90), (cw - 34, chh - 34, 0)):
        d.arc(L(x - 16, y - 16, x + 16, y + 16), a0, a0 + 270, fill=(240, 200, 110, 200), width=3 * s)
    g = (236, 196, 106, 255)
    gd = (160, 118, 40, 255)
    cx = cw / 2
    # halo
    d.ellipse(L(cx - 120, 50, cx + 120, 290), fill=(40, 80, 64, 255))
    if kind == "kunanbai":
        d.pieslice(L(cx - 140, 230, cx + 140, 470), 180, 360, fill=g)              # shoulders
        d.ellipse(L(cx - 56, 112, cx + 56, 236), fill=g)                           # head
        d.polygon(L(cx - 40, 200, cx + 40, 200, cx + 24, 268, cx, 286, cx - 24, 268), fill=g)  # beard
        d.polygon(L(cx - 70, 136, cx + 70, 136, cx + 50, 64, cx, 46, cx - 50, 64), fill=gd)     # tall hat
        d.rectangle(L(cx - 76, 128, cx + 76, 146), fill=g)
        d.line(L(cx, 290, cx, 360), fill=gd, width=5 * s)
    elif kind == "zere":
        d.pieslice(L(cx - 130, 236, cx + 130, 470), 180, 360, fill=g)
        d.ellipse(L(cx - 86, 84, cx + 86, 300), fill=(250, 240, 220, 255))        # kimeshek
        d.ellipse(L(cx - 50, 120, cx + 50, 236), fill=g)                           # face
        d.ellipse(L(cx - 82, 66, cx + 82, 150), fill=(250, 240, 220, 255))        # turban top
        d.arc(L(cx - 70, 74, cx + 70, 140), 200, 340, fill=gd, width=4 * s)
        d.arc(L(cx - 30, 196, cx + 30, 224), 20, 160, fill=gd, width=3 * s)
    else:  # ulzhan
        d.pieslice(L(cx - 130, 236, cx + 130, 470), 180, 360, fill=g)
        d.polygon(L(cx - 80, 110, cx + 80, 110, cx + 100, 300, cx - 100, 300), fill=gd)  # scarf
        d.ellipse(L(cx - 80, 60, cx + 80, 200), fill=gd)
        d.ellipse(L(cx - 48, 112, cx + 48, 232), fill=g)
        for k in range(5):
            d.ellipse(L(cx - 70 + k * 35 - 5, 280, cx - 70 + k * 35 + 5, 290), fill=g)
    return vec_done(im)

CARDS = [("kunanbai", "ҚҰНАНБАЙ", "Абайдың әкесі", 18.85), ("zere", "ЗЕРЕ", "Абайдың әжесі", 20.05),
         ("ulzhan", "ҰЛЖАН", "Абайдың анасы", 20.75)]

# ---------- yurt (draw-on)
def draw_yurt(dst, t, s0, cx, cy, wd=400, op=1.0):
    p = prog(t, s0, 0.9)
    if p <= 0:
        return
    h = wd * 0.72
    lay = np.zeros((H, W), np.float32)
    x0, x1 = cx - wd / 2, cx + wd / 2
    yb, yw = cy + h / 2, cy + h * 0.02
    k = eout(p)
    th = 5
    # base line + walls
    cv2.line(lay, (int(x0 - 20), int(yb)), (int(lerp(x0 - 20, x1 + 20, k)), int(yb)), 1, th, cv2.LINE_AA)
    wy = lerp(yb, yw, eout(prog(t, s0 + 0.15, 0.4)))
    cv2.line(lay, (int(x0), int(yb)), (int(x0), int(wy)), 1, th, cv2.LINE_AA)
    cv2.line(lay, (int(x1), int(yb)), (int(x1), int(wy)), 1, th, cv2.LINE_AA)
    # lattice
    kk = eout(prog(t, s0 + 0.3, 0.4))
    n = 9
    for i in range(n):
        xa = x0 + (x1 - x0) * i / n
        xb = x0 + (x1 - x0) * (i + 1) / n
        if i / n < kk:
            cv2.line(lay, (int(xa), int(yb)), (int(xb), int(yw)), 0.6, 2, cv2.LINE_AA)
            cv2.line(lay, (int(xb), int(yb)), (int(xa), int(yw)), 0.6, 2, cv2.LINE_AA)
    # dome
    kd = eout(prog(t, s0 + 0.45, 0.45))
    if kd > 0:
        cv2.ellipse(lay, (int(cx), int(yw)), (int(wd / 2), int(h * 0.5)), 0, 180, 180 + 180 * kd, 1, th, cv2.LINE_AA)
        for i in range(1, 8):
            ang = math.pi + math.pi * i / 8
            if i / 8 < kd:
                ex, ey = cx + wd / 2 * math.cos(ang), yw + h * 0.5 * math.sin(ang)
                cv2.line(lay, (int(cx), int(yw - h * 0.47)), (int(ex), int(ey)), 0.55, 2, cv2.LINE_AA)
        cv2.ellipse(lay, (int(cx), int(yw - h * 0.47)), (34, 12), 0, 0, 360, 1, 4, cv2.LINE_AA)
    # door
    kdoor = eout(prog(t, s0 + 0.6, 0.3))
    if kdoor > 0:
        dw, dh = wd * 0.16, h * 0.42
        cv2.rectangle(lay, (int(cx - dw / 2), int(yb - dh * kdoor)), (int(cx + dw / 2), int(yb)), 1, th, cv2.LINE_AA)
    glow = cv2.GaussianBlur(lay, (0, 0), 7)
    lay = (lay + glow * 1.3)[..., None] * op
    dark = cv2.GaussianBlur((lay[..., 0] > 0.05).astype(np.float32), (0, 0), 25)[..., None] * 0.35 * op
    dst *= (1 - dark)
    dst += lay * np.array([1.0, 0.8, 0.42], np.float32)

# ---------- animated mini-film (for "анимациялық бейнеролик")
@functools.lru_cache(None)
def hills_layer(seed, color, amp, base, w=1100, h=620):
    rng = np.random.default_rng(seed)
    xs = np.arange(w * 2)
    y = base + sum(amp / (k + 1) * np.sin(xs / (w / (k + 1.3)) * 2 * math.pi + rng.uniform(0, 6)) for k in range(4))
    img = np.zeros((h, w * 2, 4), np.float32)
    yy = np.arange(h)[:, None]
    m = (yy >= y[None, :]).astype(np.float32)
    img[..., :3] = np.array(color, np.float32) / 255
    img[..., 3] = m
    return img

def mini_film(t, t0, w=1100, h=620):
    tt = t - t0
    sky = np.zeros((h, w, 3), np.float32)
    p = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top = np.array([0.20, 0.16, 0.38], np.float32); mid = np.array([0.95, 0.55, 0.30], np.float32)
    sky[:] = top * (1 - p) + mid * p
    # sun
    sx, sy = int(w * 0.68), int(h * 0.62 - tt * 18)
    g = np.zeros((h, w), np.float32)
    cv2.circle(g, (sx, sy), 60, 1, -1, cv2.LINE_AA)
    sky += cv2.GaussianBlur(g, (0, 0), 30)[..., None] * np.array([1.0, 0.7, 0.35], np.float32) * 0.9
    sky += g[..., None] * np.array([1.0, 0.92, 0.7], np.float32) * 0.6
    for seed, col, amp, base, sp in ((1, (120, 70, 90), 50, 360, 20), (2, (70, 40, 60), 40, 430, 45), (3, (30, 22, 34), 30, 500, 90)):
        L = hills_layer(seed, col, amp, base, w, h)
        off = int(tt * sp) % w
        over(sky, L[:, off:off + w], 0, 0)
    # yurts silhouettes
    for (x, sc) in ((int(w * 0.25 - tt * 90) % (w + 200) - 100, 1.0), (int(w * 0.42 - tt * 90) % (w + 200) - 100, 0.8)):
        yb = 545
        cv2.rectangle(sky, (x - int(45 * sc), yb - int(40 * sc)), (x + int(45 * sc), yb), (0.08, 0.05, 0.08), -1)
        cv2.ellipse(sky, (x, yb - int(40 * sc)), (int(45 * sc), int(30 * sc)), 0, 180, 360, (0.08, 0.05, 0.08), -1, cv2.LINE_AA)
    # rider
    rx = int(-120 + tt * 330)
    ry = 520 + int(6 * math.sin(tt * 18))
    col = (0.05, 0.03, 0.05)
    cv2.ellipse(sky, (rx, ry), (52, 22), 0, 0, 360, col, -1, cv2.LINE_AA)
    cv2.circle(sky, (rx + 54, ry - 26), 14, col, -1, cv2.LINE_AA)
    cv2.line(sky, (rx + 40, ry - 10), (rx + 58, ry - 26), col, 14, cv2.LINE_AA)
    for k in range(4):
        ph = tt * 18 + k * 1.6
        lx = rx - 34 + k * 22
        cv2.line(sky, (lx, ry + 12), (lx + int(10 * math.sin(ph)), ry + 52), col, 7, cv2.LINE_AA)
    cv2.circle(sky, (rx + 4, ry - 50), 11, col, -1, cv2.LINE_AA)
    cv2.line(sky, (rx + 2, ry - 40), (rx + 6, ry - 14), col, 12, cv2.LINE_AA)
    # title in film
    if tt > 1.3:
        k = eout(prog(t, t0 + 1.3, 0.5))
        text_draw(sky, "АБАЙ ЖОЛЫ", w / 2, 170 + (1 - k) * 40, 96, op=k, fname="Black",
                  grad=(GOLD_L, GOLD), extrude=5, ext_col=(80, 45, 10), shadow=0.6)
    # film UI
    cv2.rectangle(sky, (40, h - 40), (w - 40, h - 34), (1, 1, 1), -1)
    cv2.rectangle(sky, (40, h - 40), (40 + int((w - 80) * clamp(tt / 2.6)), h - 34), (1.0, 0.8, 0.4), -1)
    rgba = np.dstack([np.clip(sky, 0, 1), rounded_mask(w, h, 26)])
    # border
    e = np.clip(rgba[..., 3] - cv2.erode(rgba[..., 3], np.ones((7, 7), np.uint8)), 0, 1)[..., None]
    rgba[..., :3] = rgba[..., :3] * (1 - e) + np.array([1.0, 0.85, 0.5], np.float32) * e
    return rgba

# ------------------------------------------------------------------ face window (pip)
def pip_geom(t, si, seg):
    s, e, _, cfg = seg
    x, y, w, h = cfg["rect"]
    pin = eout(prog(t, s, TR))
    pout = ein(prog(t, e - TR, TR))
    # if the neighbouring segment is bg (not full), still animate from/to full frame
    p = pin * (1 - pout)
    rx = lerp(0, x, p); ry = lerp(0, y, p); rw = lerp(W, w, p); rh = lerp(H, h, p)
    rr = lerp(0, cfg["r"], p)
    return p, (rx, ry, rw, rh, rr)

def face_window(dst, frame, t, geom, p):
    rx, ry, rw, rh, rr = geom
    rw_i, rh_i = int(round(rw)), int(round(rh))
    f = face_at(t)
    fh = f["y1"] - f["y0"]
    # target crop: face-centred with aspect of window
    asp = rw / rh
    ch_t = min(H, fh * 1.75)
    cw_t = ch_t * asp
    if cw_t > W:
        cw_t = W; ch_t = W / asp
    fx, fy = face_slow(t)
    ccx_t, ccy_t = fx, fy + fh * 0.18
    # full frame crop with same aspect
    ch_f = H if asp <= W / H else W / asp
    cw_f = ch_f * asp
    chh = lerp(ch_f, ch_t, p); cww = chh * asp
    ccx = lerp(W / 2, ccx_t, p); ccy = lerp(H / 2, ccy_t, p)
    ccx = clamp(ccx, cww / 2, W - cww / 2); ccy = clamp(ccy, chh / 2, H - chh / 2)
    x0, y0 = ccx - cww / 2, ccy - chh / 2
    M = np.float32([[rw_i / cww, 0, -x0 * rw_i / cww], [0, rh_i / chh, -y0 * rh_i / chh]])
    crop = cv2.warpAffine(frame, M, (rw_i, rh_i), flags=cv2.INTER_LINEAR)
    m = rounded_mask(rw_i, rh_i, rr)[..., None] if rr > 0.5 else np.ones((rh_i, rw_i, 1), np.float32)
    X, Y = int(round(rx)), int(round(ry))
    # shadow + gold border
    if p > 0.02:
        pad = 40
        sh = np.zeros((rh_i + 2 * pad, rw_i + 2 * pad), np.float32)
        sh[pad:pad + rh_i, pad:pad + rw_i] = m[..., 0]
        sh = cv2.GaussianBlur(sh, (0, 0), 18)[..., None] * 0.75 * p
        y0s, x0s = Y - pad + 14, X - pad
        ys0, xs0 = max(0, y0s), max(0, x0s)
        ys1, xs1 = min(H, y0s + sh.shape[0]), min(W, x0s + sh.shape[1])
        dst[ys0:ys1, xs0:xs1] *= (1 - sh[ys0 - y0s:ys1 - y0s, xs0 - x0s:xs1 - x0s])
    rgba = np.dstack([crop, m[..., 0]])
    over(dst, rgba, X, Y)
    if p > 0.02:
        e = np.clip(m[..., 0] - cv2.erode(m[..., 0], np.ones((7, 7), np.uint8)), 0, 1)[..., None] * p
        reg = dst[Y:Y + rh_i, X:X + rw_i]
        hh, ww = reg.shape[:2]
        e = e[:hh, :ww]
        reg *= (1 - e)
        reg += e * np.array([1.0, 0.83, 0.45], np.float32)

# ------------------------------------------------------------------ per-segment graphics
def g_intro(dst, t, face, z, px, py):
    # face-tracked labels
    ax, ay = cam_pt(face["x0"] + 10, face["cy"] - 40, z, px, py)
    label_tag(dst, t, 0.3, 3.5, "ҚАЗАҚ ӘДЕБИЕТІ", "классика", ax - 70, ay - 120, "l", anchor_pt=(ax - 20, ay))
    bx, by = cam_pt(face["x1"] - 10, face["cy"] - 20, z, px, py)
    label_tag(dst, t, 3.65, 4.75, "МҰХТАР ӘУЕЗОВ", "жазушы, 1897–1961", bx + 150, by - 110, "r", anchor_pt=(bx + 20, by))

def g_title(dst, t, s0=4.70, end=7.05):
    p = prog(t, s0, 0.75)
    cov = framed("cover.png", 600)
    ry = lerp(-85, -16, eback(p, 1.2)) + 4 * math.sin(t * 1.3)
    sc = lerp(0.55, 1.0, eout(p))
    op = eout(prog(t, s0, 0.25)) * (1 - ein(prog(t, end - 0.15, 0.15)))
    warp(dst, cov, 470, 400, ry=ry, rx=3 * math.sin(t), s=sc, op=op)
    text_draw(dst, "МҰХТАР ӘУЕЗОВ", 1300, 175, 40, op=op * eout(prog(t, s0 + 0.1, 0.4)), fname="Bold", track=10,
              fill=(235, 225, 200))
    title = "АБАЙ ЖОЛЫ"
    f = font("Black", 128)
    tw_ = text_w(title, f, 4)
    x = 1300 - tw_ / 2
    for i, c in enumerate(title):
        cw = f.getlength(c)
        if c != " ":
            q = prog(t, s0 + 0.1 + i * 0.06, 0.5)
            text_draw(dst, c, x + cw / 2, 320 + (1 - eout(q)) * -60, 128, op=op * eout(q), rx=(1 - eback(q)) * -90,
                      fname="Black", grad=((255, 240, 190), (212, 148, 38)), extrude=10, ext_col=(95, 58, 12),
                      shadow=0.7, glow=((255, 190, 80), 0.25))
        x += cw + 4
    q = prog(t, s0 + 0.7, 0.8)
    if q > 0:
        tr = int(lerp(2, 22, eout(q)))
        text_draw(dst, "РОМАН-ЭПОПЕЯ", 1300, 460, 46, op=op * eout(q), fname="Medium", track=tr, fill=(255, 240, 210))
        lw = int(260 * eout(q))
        cv2.line(dst, (1300 - lw, 520), (1300 + lw, 520), (1.0, 0.82, 0.45), 2, cv2.LINE_AA)
    gold_burst(dst, t, s0 + 0.15, 1300, 320, 80, 520)

def g_volume(dst, t, s0=7.05, end=10.05):
    book = asset("book3d.png", height=500)
    p = prog(t, s0 + 0.2, 0.9)
    op = eout(prog(t, s0 + 0.2, 0.3)) * (1 - ein(prog(t, end - 0.35, 0.3)))
    ry = lerp(160, -18, eout(p)) + 6 * math.sin(t * 1.5)
    # glow
    gl = np.zeros((H, W), np.float32)
    cv2.circle(gl, (1230, 410), 200, 1, -1)
    gl = cv2.GaussianBlur(gl, (0, 0), 70)[..., None] * 0.45 * op
    dst += gl * np.array([1.0, 0.75, 0.3], np.float32)
    warp(dst, book, 1230, 410 + 8 * math.sin(t * 2), ry=ry, s=lerp(0.7, 1.0, eout(p)), op=op)
    q = prog(t, 8.75, 0.5)
    if q > 0:
        text_draw(dst, "БІРІНШІ", 1610, 300, 40, op=op * eout(q), fname="Bold", track=8, fill=(255, 236, 190))
        text_draw(dst, "I", 1610, 430, 210, op=op * eout(q), s=lerp(1.6, 1.0, eout(q)), ry=(1 - eout(q)) * 80,
                  fname="/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", grad=((255, 240, 190), (212, 148, 38)),
                  extrude=10, ext_col=(95, 58, 12), shadow=0.7, glow=((255, 190, 80), 0.3))
        text_draw(dst, "ТОМ", 1610, 580, 64, op=op * eout(prog(t, 8.95, 0.4)), fname="Black", track=10, fill=(255, 255, 255))
        gold_burst(dst, t, 8.8, 1610, 430, 60, 330)

def g_abai(dst, t, s0=10.3, end=12.05):
    vis = window(t, s0, end, 0.35, 0.3)
    if vis <= 0:
        return
    p = prog(t, s0, 0.6)
    a = asset("abai.png", height=470)
    warp(dst, a, 330, 380, ry=lerp(55, 0, eout(p)) + 3 * math.sin(t * 2), s=lerp(0.6, 1.0, eback(p)) * (1 - 0.3 * ein(prog(t, end - 0.3, 0.3))), op=vis)
    cv2.ellipse(dst, (330, 380), (int(176 * eout(p)), int(236 * eout(p))), 0, -90, -90 + 360 * eout(p), (1.0, 0.82, 0.45), 3, cv2.LINE_AA)
    text_draw(dst, "АБАЙ ҚҰНАНБАЙҰЛЫ", 330, 660, 34, op=vis * eout(prog(t, s0 + 0.3, 0.3)), fname="ExtraBold", track=4,
              fill=(255, 236, 190))
    text_draw(dst, "1845–1904", 330, 702, 26, op=vis * eout(prog(t, s0 + 0.4, 0.3)) * 0.85, fname="Medium", fill=(230, 230, 230))

LIST = [("child", "Балалық шақ", 14.05), ("yurt", "Өскен ортасы", 14.25), ("family", "Отбасылық тәрбие", 14.45),
        ("eye", "Өмірге көзқарас", 16.15)]

def g_list(dst, t, end=17.95):
    for i, (ic, txt, s) in enumerate(LIST):
        p = prog(t, s, 0.5)
        if p <= 0:
            continue
        op = eout(p) * (1 - ein(prog(t, end - 0.35, 0.3)))
        y = 185 + i * 130
        b = icon_badge(ic)
        warp(dst, b, 220, y, ry=(1 - eout(p)) * -90, s=lerp(0.5, 1.0, eback(p)), op=op)
        text_draw(dst, txt, 310 + (1 - eout(p)) * -40, y, 50, op=op, anchor="l", fname="ExtraBold", fill=(255, 255, 255))
        # active highlight: the newest item
        if i == 3 or (t < 16.15 and i == 2):
            k = window(t, s, end, 0.3, 0.3) * 0.5
            cv2.line(dst, (310, int(y + 38)), (310 + int(380 * eout(prog(t, s + 0.2, 0.5))), int(y + 38)), (1.0, 0.8, 0.4), 3, cv2.LINE_AA)

def g_cards(dst, t, s0=17.95, end=22.05):
    op_all = eout(prog(t, s0, 0.3)) * (1 - ein(prog(t, end - 0.3, 0.3)))
    text_draw(dst, "КЕЙІПКЕРЛЕР", 960, 92, 44, op=op_all * eout(prog(t, s0 + 0.1, 0.4)), fname="ExtraBold", track=14,
              fill=(255, 236, 190))
    cv2.line(dst, (960 - int(170 * eout(prog(t, s0 + 0.2, 0.5))), 128), (960 + int(170 * eout(prog(t, s0 + 0.2, 0.5))), 128),
             (1.0, 0.82, 0.45), 2, cv2.LINE_AA)
    xs = [530, 960, 1390]
    for i, (k, name, role, ts) in enumerate(CARDS):
        p = prog(t, ts - 0.15, 0.6)
        if p <= 0:
            continue
        cd = char_card(k)
        active = (i == 2 and t >= CARDS[2][3]) or (i < 2 and CARDS[i][3] <= t < CARDS[i + 1][3])
        bright = 1.0 if active else 0.75
        y = 440 + 6 * math.sin(t * 2 + i)
        warp(dst, cd, xs[i], y, ry=lerp(-100, 0, eback(p, 1.3)) + (2 * math.sin(t * 1.5 + i)), s=1.0 + (0.04 if active else 0),
             op=op_all * eout(prog(t, ts - 0.15, 0.2)) * bright)
        q = prog(t, ts + 0.1, 0.4)
        text_draw(dst, name, xs[i], y + 150, 44, op=op_all * eout(q), fname="Black", track=3,
                  grad=(GOLD_L, (214, 150, 40)), shadow=0.6)
        text_draw(dst, role, xs[i], y + 196, 26, op=op_all * eout(q) * 0.9, fname="Medium", fill=(235, 235, 235))

def g_society(dst, t, face, z, px, py):
    # network icon + label (left), 22.4–24.6
    vis = window(t, 22.45, 24.7, 0.3, 0.3)
    if vis > 0:
        b = icon_badge("network", 150)
        p = prog(t, 22.45, 0.5)
        warp(dst, b, 300, 330, ry=(1 - eout(p)) * 80, s=lerp(0.5, 1, eback(p)), op=vis)
        text_draw(dst, "ӘЛЕУМЕТТІК", 300, 470, 38, op=vis * eout(prog(t, 22.6, 0.3)), fname="ExtraBold", track=4, fill=(255, 236, 190))
        text_draw(dst, "ҚАТЫНАСТАР", 300, 514, 38, op=vis * eout(prog(t, 22.7, 0.3)), fname="ExtraBold", track=4, fill=(255, 236, 190))
    # traditions: yurt left + dombra right
    s0, end = 24.85, 27.0
    vis = window(t, s0, end, 0.3, 0.3)
    if vis > 0:
        draw_yurt(dst, t, s0, 300, 400, 380, op=vis)
        text_draw(dst, "САЛТ-ДӘСТҮР", 300, 600, 40, op=vis * eout(prog(t, s0 + 0.5, 0.3)), fname="ExtraBold", track=6,
                  fill=(255, 236, 190))
        dmb = asset("dombra.png", height=640)
        p = prog(t, s0 + 0.15, 0.7)
        warp(dst, dmb, 1640, 400 + (1 - eout(p)) * 300, rz=lerp(-40, -16, eback(p)) + 2 * math.sin(t * 2),
             ry=lerp(80, 10, eout(p)), s=1.0, op=vis)

def g_era(dst, t, s0=27.05, end=31.7):
    op_all = 1 - ein(prog(t, end - 0.35, 0.3))
    # one person
    p = prog(t, 28.55, 0.5)
    if p > 0:
        pi = icon("person", 150)
        warp(dst, pi, 900, 300, s=lerp(0.4, 1.0, eback(p)), op=eout(p) * op_all)
        text_draw(dst, "БІР АДАМ", 900, 440, 50, op=eout(p) * op_all, fname="Black", track=4, fill=(255, 255, 255))
    # arrow
    q = prog(t, 29.6, 0.7)
    if q > 0:
        x0, x1 = 1020, int(lerp(1020, 1240, eout(q)))
        cv2.line(dst, (x0, 300), (x1, 300), (1.0, 0.82, 0.45), 5, cv2.LINE_AA)
        if q > 0.6:
            cv2.fillConvexPoly(dst, np.array([[x1 + 22, 300], [x1 - 6, 284], [x1 - 6, 316]], np.int32), (1.0, 0.82, 0.45), cv2.LINE_AA)
    # crowd = era
    r = prog(t, 30.3, 0.9)
    if r > 0:
        pi = icon("person", 64)
        k = 0
        for row in range(3):
            for col in range(5):
                d = prog(t, 30.3 + k * 0.035, 0.4)
                warp(dst, pi, 1360 + col * 70, 215 + row * 72, s=eback(d), op=eout(d) * op_all * (0.6 + 0.4 * (row == 1)))
                k += 1
        text_draw(dst, "ТҰТАС ДӘУІР", 1500, 470, 64, op=eout(prog(t, 30.45, 0.4)) * op_all, s=lerp(1.4, 1, eout(prog(t, 30.45, 0.4))),
                  fname="Black", grad=((255, 240, 190), (212, 148, 38)), extrude=6, ext_col=(95, 58, 12), shadow=0.7,
                  glow=((255, 190, 80), 0.3))
        gold_burst(dst, t, 30.5, 1500, 300, 60, 420)

def g_era_label(dst, t, face, z, px, py):
    bx, by = cam_pt(face["x1"] - 10, face["cy"] - 30, z, px, py)
    label_tag(dst, t, 31.8, 33.1, "XIX ҒАСЫР", "қазақ даласы", bx + 150, by - 100, "r", anchor_pt=(bx + 20, by))

def g_film(dst, t, s0=34.3, end=36.95):
    p = prog(t, s0, 0.7)
    op = eout(prog(t, s0, 0.25)) * (1 - ein(prog(t, end - 0.2, 0.2)))
    mf = mini_film(t, s0)
    warp(dst, mf, 960, 410, ry=lerp(70, -10, eout(p)) + 3 * math.sin(t), rx=lerp(20, 4, eout(p)), s=lerp(0.6, 0.98, eout(p)), op=op)
    tag = "АНИМАЦИЯЛЫҚ ЖАРНАМА"
    text_draw(dst, tag, 960, 66, 36, op=op * eout(prog(t, s0 + 0.3, 0.4)), fname="ExtraBold", track=10, fill=(255, 236, 190))

def g_tech(dst, t, s0=39.25, end=42.65):
    op_all = 1 - ein(prog(t, end - 0.35, 0.3))
    for (ic, label, ts, x) in (("book", "ӘДЕБИ МҰРА", 39.3, 250), ("chip", "ЖАҢА ТЕХНОЛОГИЯ", 40.55, 700)):
        p = prog(t, ts, 0.55)
        if p <= 0:
            continue
        b = icon_badge(ic, 200)
        warp(dst, b, x, 360 + 6 * math.sin(t * 2 + x), ry=(1 - eout(p)) * 100, s=lerp(0.4, 1.0, eback(p)), op=eout(p) * op_all)
        text_draw(dst, label, x, 520, 36, op=eout(prog(t, ts + 0.2, 0.3)) * op_all, fname="ExtraBold", track=3, fill=(255, 236, 190))
    q = prog(t, 40.25, 0.35)
    if q > 0:
        text_draw(dst, "+", 475, 360, 110, op=eout(q) * op_all, s=eback(q), fname="Black", fill=(255, 210, 120), shadow=0.5)
    gold_burst(dst, t, 40.6, 700, 360, 60, 300)

def g_bridge(dst, t, face, z, px, py):
    s0, end = 42.75, 45.9
    vis = window(t, s0, end, 0.3, 0.3)
    if vis <= 0:
        return
    L = cam_pt(face["x0"] - 30, face["cy"] - 60, z, px, py)
    R = cam_pt(face["x1"] + 30, face["cy"] - 60, z, px, py)
    lx, rx = L[0] - 230, R[0] + 230
    yy = L[1]
    p1 = prog(t, s0, 0.4); p2 = prog(t, 43.3, 0.4)
    text_draw(dst, "ӨТКЕН", lx, yy, 64, op=vis * eout(p1), s=lerp(0.6, 1, eback(p1)), ry=(1 - eout(p1)) * 70, fname="Black",
              grad=((255, 240, 190), (212, 148, 38)), extrude=5, ext_col=(95, 58, 12), shadow=0.8)
    text_draw(dst, "БҮГІН", rx, yy, 64, op=vis * eout(p2), s=lerp(0.6, 1, eback(p2)), ry=(1 - eout(p2)) * -70, fname="Black",
              fill=(255, 255, 255), extrude=5, ext_col=(60, 60, 70), shadow=0.8)
    a = prog(t, 43.8, 0.9)
    if a > 0:
        top = cam_pt(face["cx"], face["top"], z, px, py)[1]
        peak = max(30, top - 40)
        n = 60
        pts = []
        for i in range(int(n * eout(a)) + 1):
            u = i / n
            x = lerp(lx, rx, u)
            y = (1 - u) ** 2 * (yy - 50) + 2 * (1 - u) * u * (peak - 60) + u ** 2 * (yy - 50)
            pts.append((int(x), int(y)))
        if len(pts) > 1:
            lay = np.zeros((H, W), np.float32)
            cv2.polylines(lay, [np.array(pts, np.int32)], False, 1.0, 4, cv2.LINE_AA)
            lay = (lay + cv2.GaussianBlur(lay, (0, 0), 6) * 1.5)[..., None] * vis
            dst += lay * np.array([1.0, 0.8, 0.4], np.float32)
            hx, hy = pts[-1]
            cv2.circle(dst, (hx, hy), 9, (1.0, 0.95, 0.8), -1, cv2.LINE_AA)

def g_end(dst, t, s0=49.0):
    p = prog(t, s0, 0.6)
    cov = framed("cover.png", 560)
    warp(dst, cov, 560, 420, ry=lerp(-70, -14, eout(p)) + 3 * math.sin(t), s=lerp(0.7, 1, eout(p)), op=eout(prog(t, s0, 0.25)))
    q = prog(t, s0 + 0.15, 0.5)
    text_draw(dst, "АБАЙ ЖОЛЫ", 1270, 360, 130, op=eout(q), s=lerp(1.3, 1, eout(q)), fname="Black",
              grad=((255, 240, 190), (212, 148, 38)), extrude=9, ext_col=(95, 58, 12), shadow=0.7, glow=((255, 190, 80), 0.25))
    text_draw(dst, "I ТОМ  ·  МҰХТАР ӘУЕЗОВ", 1270, 490, 40, op=eout(prog(t, s0 + 0.35, 0.4)), fname="Bold", track=6, fill=(255, 240, 210))
    gold_burst(dst, t, s0 + 0.2, 1270, 360, 80, 520)

# ------------------------------------------------------------------ frame
MODE_SWITCH = [4.70, 7.05, 17.95, 22.05, 34.30, 36.95, 49.0]

def render_frame(t, src_u8):
    i, seg = seg_at(t)
    mode = seg[2]
    face = face_at(t)
    src = src_u8.astype(np.float32) / 255.0
    z, px, py = cam(t)
    behind = next((b for b in BEHIND if b[0] <= t < b[1]), None)
    if mode == "full":
        out = cam_apply(src, z, px, py)
        if behind:
            alpha = person_alpha(src_u8)
            alpha = cam_apply(alpha[..., 0], z, px, py)[..., None]
            person = out.copy()
            draw_behind(out, t, behind, face, z, px, py)
            out = out * (1 - alpha) + person * alpha
    elif mode == "bg":
        out = draw_bg(t)
    else:  # pip
        out = draw_bg(t, orn_xy=(seg[3]["rect"][0] + seg[3]["rect"][2] / 2, seg[3]["rect"][1] + seg[3]["rect"][3] / 2), orn_op=0.10)
        p, geom = pip_geom(t, i, seg)
        face_window(out, src, t, geom, p)
    # graphics
    if mode == "full":
        if t < 4.75: g_intro(out, t, face, z, px, py)
        if 10.0 < t < 12.1: g_abai(out, t)
        if 22.0 < t < 27.1: g_society(out, t, face, z, px, py)
        if 31.7 < t < 33.2: g_era_label(out, t, face, z, px, py)
        if 42.6 < t < 46.0: g_bridge(out, t, face, z, px, py)
    elif mode == "bg":
        if 4.6 < t < 7.1: g_title(out, t)
        if 17.9 < t < 22.1: g_cards(out, t)
        if 34.2 < t < 37.0: g_film(out, t)
        if t >= 49.0: g_end(out, t)
    else:
        if 7.0 < t < 10.1: g_volume(out, t)
        if 13.9 < t < 18.0: g_list(out, t)
        if 27.0 < t < 31.8: g_era(out, t)
        if 39.2 < t < 42.7: g_tech(out, t)
    # bottom gradient + subtitles
    out *= (1 - bottom_grad())
    if mode == "full":
        chin = max(face_at(t + d)["y1"] for d in (-0.3, 0, 0.3, 0.6))
        chin_o = cam_pt(0, chin, z, px, py)[1]
        yc = clamp(max(790, chin_o + 78), 790, 985)
    else:
        yc = 820
    draw_subs(out, t, subs_y(t, yc))
    flash(out, t, MODE_SWITCH)
    # cinematic finishing
    out *= vignette()
    out += grain(int(t * FPS) % 6)
    # fade at very end
    out *= 1 - 0.0 * prog(t, DUR - 0.3, 0.3)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

_SUBY = {}
def subs_y(t, yc):
    # keep subtitle line stable per chunk: use max over chunk lifetime (computed lazily)
    for k, c in enumerate(CH):
        if c["show"] <= t < c["hide"]:
            if k not in _SUBY:
                ys = []
                for tt in np.arange(c["show"], c["hide"], 0.1):
                    ii, sg = seg_at(tt)
                    if sg[2] == "full":
                        f = face_at(tt); z, px, py = cam(tt)
                        chin = max(face_at(tt + d)["y1"] for d in (-0.3, 0, 0.3))
                        ys.append(clamp(max(790, cam_pt(0, chin, z, px, py)[1] + 78), 790, 985))
                    else:
                        ys.append(820)
                _SUBY[k] = max(ys) if ys else 820
            return _SUBY[k]
    return yc

# ------------------------------------------------------------------ IO
def init():
    global CH
    load_faces()
    CH = build_chunks()

def read_frames(path, start=0, count=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H * 3
    i = 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        if i >= start and (count is None or i < start + count):
            yield i, np.frombuffer(b, np.uint8).reshape(H, W, 3)
        i += 1
        if count is not None and i >= start + count:
            break
    p.stdout.close(); p.kill()

def grab(t):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.3f" % t, "-i", GRADED, "-frames:v", "1", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    return np.frombuffer(p.stdout[:W * H * 3], np.uint8).reshape(H, W, 3)

def worker(args):
    a, b, out = args
    init()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                            "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", out],
                           stdin=subprocess.PIPE)
    last = None
    for i, fr in read_frames(GRADED, a, b - a):
        last = fr
        enc.stdin.write(render_frame(i / FPS, fr).tobytes())
        if i % 60 == 0:
            print("frame", i, flush=True)
    enc.stdin.close(); enc.wait()
    return out

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sketch":
        init()
        outdir = sys.argv[2]
        for ts in sys.argv[3:]:
            t = float(ts)
            img = render_frame(t, grab(t))
            Image.fromarray(img).save("%s/f_%05.2f.jpg" % (outdir, t), quality=90)
            print("ok", t)
    elif cmd == "render":
        from multiprocessing import Pool
        out = sys.argv[2]
        nf = int(sys.argv[3]) if len(sys.argv) > 3 else NFR
        nw = 4
        bounds = [int(nf * k / nw) for k in range(nw + 1)]
        parts = ["%s.part%d.mp4" % (out, k) for k in range(nw)]
        with Pool(nw) as pool:
            pool.map(worker, [(bounds[k], bounds[k + 1], parts[k]) for k in range(nw)])
        with open(out + ".txt", "w") as fh:
            for p_ in parts:
                fh.write("file '%s'\n" % os.path.abspath(p_))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", out + ".txt", "-c", "copy", out], check=True)
        print("done", out)
