# synth sound design for Абай жолы — writes sfx.wav (stereo 48k)
import numpy as np, wave, sys, os
SR = 48000
DUR = 50.245
rng = np.random.default_rng(11)
N = int(DUR * SR) + SR
L = np.zeros(N, np.float32); R = np.zeros(N, np.float32)

def db(x): return 10 ** (x / 20)
def t_(d): return np.arange(int(d * SR)) / SR
def norm(x): return x / (np.max(np.abs(x)) + 1e-9)

def lp(x, a):
    y = np.zeros_like(x); s = 0.0
    a = np.broadcast_to(a, x.shape)
    for i in range(len(x)):
        s += a[i] * (x[i] - s); y[i] = s
    return y

def whoosh(d=0.5, up=True):
    n = int(d * SR); x = rng.standard_normal(n)
    c = np.linspace(0.02, 0.25, n) if up else np.linspace(0.25, 0.02, n)
    y = lp(x, c) - lp(x, c * 0.15)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return norm(y * env)

def impact(d=1.2):
    t = t_(d)
    f = 45 + 90 * np.exp(-t * 18)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    click = lp(rng.standard_normal(len(t)), 0.3) * np.exp(-t * 40)
    return norm(body + 0.5 * click)

def pop():
    t = t_(0.14); f = 700 * np.exp(-t * 12) + 300
    return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.002) * np.exp(-t * 30))

def tick():
    t = t_(0.04); x = rng.standard_normal(len(t)) * np.exp(-t * 180)
    return norm(np.convolve(x, [1, -0.95], "same"))

def shimmer(d=0.9):
    t = t_(d); y = np.zeros(len(t))
    for k in range(14):
        st = rng.uniform(0, d * 0.6); f = rng.uniform(2400, 6200)
        m = t >= st; tt = t[m] - st
        y[m] += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 9) * rng.uniform(0.4, 1)
    return norm(y * np.minimum(1, t / 0.01))

def riser(d=0.8):
    t = t_(d); x = rng.standard_normal(len(t))
    y = lp(x, np.linspace(0.01, 0.35, len(t))) * (t / d) ** 2
    s = np.sin(2 * np.pi * np.cumsum(200 + 900 * (t / d) ** 2) / SR) * (t / d) ** 3 * 0.3
    return norm(y + s)

def pluck(f=196, d=1.4):
    n = int(d * SR); p = int(SR / f); buf = rng.uniform(-1, 1, p); y = np.zeros(n)
    for i in range(n):
        y[i] = buf[i % p]; buf[i % p] = 0.996 * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return norm(y)

def ding(f=1320):
    t = t_(0.8)
    return norm((np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.76 * t)) * np.exp(-t * 6) * np.minimum(1, t / 0.003))

def key():
    t = t_(0.05); x = rng.standard_normal(len(t)) * np.exp(-t * 120)
    return norm(lp(x, 0.5) + 0.5 * np.sin(2 * np.pi * 2200 * t) * np.exp(-t * 200))

def add(sig, t0, gain_db, pan=0.0):
    i = int(max(0, t0) * SR); s = sig * db(gain_db)
    j = min(N, i + len(s)); s = s[:j - i]
    L[i:j] += s * (1 - max(0, pan)); R[i:j] += s * (1 + min(0, pan))

W, IMP, POP, TICK, SH, RIS, DING, KEY = whoosh, impact, pop, tick, shimmer, riser, ding, key
# into full-background scenes: riser -> whoosh -> impact
for T in (4.70, 17.95, 34.30, 49.0):
    add(riser(0.7), T - 0.7, -26)
    add(whoosh(0.45), T - 0.25, -21)
    add(impact(), T, -15)
# back to face / pip transitions
for T in (7.05, 10.0, 13.95, 17.5, 22.05, 27.05, 31.25, 36.95, 39.25, 42.2):
    add(whoosh(0.4, up=T in (7.05, 13.95, 27.05, 39.25)), T - 0.15, -22, pan=0.0)
# subtitle phrase ticks
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for T in (0.0, 0.9, 1.9, 3.8, 7.2, 8.8, 10.1, 11.5, 12.9, 14.2, 15.3, 17.0, 18.0, 20.1, 22.0, 23.3, 24.5, 25.8, 27.0, 28.7,
          30.5, 31.5, 33.2, 34.0, 35.2, 36.9, 39.0, 40.6, 42.3, 43.8, 45.85, 47.2):
    add(tick(), T, -32)
# accents
for T in (4.8, 8.8, 18.9, 30.5, 36.9, 42.3):
    add(pop(), T, -24); add(shimmer(0.8), T, -30)
# labels (face tracked)
for T in (0.3, 3.65, 31.8, 42.75, 43.3):
    add(whoosh(0.3), T - 0.05, -27); add(pop(), T + 0.2, -26)
# title letters
for k in range(8): add(tick(), 4.8 + k * 0.06, -26)
add(shimmer(1.2), 5.4, -28)
# volume scene
add(whoosh(0.7), 7.25, -22); add(impact(0.8), 8.8, -22); add(shimmer(0.9), 8.85, -27)
# Abai medallion
add(whoosh(0.35), 10.25, -25); add(ding(990), 10.5, -30)
# behind text БАЛАЛЫҚ
add(whoosh(0.6), 11.7, -20); add(impact(0.7), 11.95, -21)
# checklist
for T in (14.05, 14.25, 14.45, 16.15): add(pop(), T, -24); add(ding(1480), T + 0.05, -33)
# character cards
for T in (18.7, 19.9, 20.6): add(whoosh(0.3), T, -24); add(pop(), T + 0.25, -24)
# society network
add(pop(), 22.45, -24); add(shimmer(0.5), 22.5, -32)
# yurt + dombra pluck
add(shimmer(1.0), 24.85, -29); add(whoosh(0.5), 25.0, -24)
for k, f in enumerate((196, 262, 196, 294)): add(pluck(f, 1.2), 25.25 + k * 0.17, -27, pan=0.4)
# era
add(pop(), 28.55, -24); add(whoosh(0.5), 29.6, -26)
for k in range(15): add(tick(), 30.3 + k * 0.035, -30)
add(impact(0.9), 30.45, -20)
# film
add(shimmer(1.4), 35.6, -30)
# CLAUDE CODE typing
for k in range(11): add(key(), 36.95 + k * 0.06, -24, pan=rng.uniform(-0.3, 0.3))
add(impact(0.6), 37.65, -24)
# tech
add(pop(), 39.3, -24); add(pop(), 40.25, -26); add(pop(), 40.55, -24); add(shimmer(0.8), 40.6, -29)
# arc
add(shimmer(1.2), 43.8, -28)
# end card
add(shimmer(1.2), 49.2, -27)

mx = max(np.max(np.abs(L)), np.max(np.abs(R)))
print("peak dBFS", 20 * np.log10(mx + 1e-9))
st = np.stack([L, R], 1)[:int(DUR * SR)]
pcm = (np.clip(st, -1, 1) * 32767).astype("<i2")
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
