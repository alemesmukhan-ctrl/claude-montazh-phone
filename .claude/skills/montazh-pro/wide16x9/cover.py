import sys, numpy as np, cv2
sys.path.insert(0, "."); import render as R
from PIL import Image
R.init()
fr = R.grab(10.0)
src = fr.astype(np.float32) / 255
a = R.person_alpha(fr)
out = R.draw_bg(3.0, orn_xy=(1330, 470), orn_op=0.16)
# warm glow behind person
gl = np.zeros((R.H, R.W), np.float32); cv2.circle(gl, (1330, 420), 330, 1, -1)
out += cv2.GaussianBlur(gl, (0, 0), 120)[..., None] * np.array([0.5, 0.36, 0.14], np.float32)
# person scaled 1.12 & shifted right
M = np.float32([[1.12, 0, 520 - 0.12 * 960], [0, 1.12, -10]])
ps = cv2.warpAffine(src, M, (R.W, R.H)); pa = cv2.warpAffine(a[..., 0], M, (R.W, R.H))[..., None]
# rim light: gold edge
edge = np.clip(pa[..., 0] - cv2.erode(pa[..., 0], np.ones((9, 9), np.uint8)), 0, 1)
out = out * (1 - pa) + ps * pa
out += cv2.GaussianBlur(edge, (0, 0), 4)[..., None] * np.array([0.9, 0.65, 0.25], np.float32) * 0.6
out *= (1 - R.bottom_grad() * 0.8)
R.text_draw(out, "МҰХТАР ӘУЕЗОВ", 560, 210, 46, fname="Bold", track=12, fill=(240, 230, 205))
R.text_draw(out, "АБАЙ", 560, 380, 210, fname="Black", grad=((255, 242, 196), (212, 148, 38)), extrude=12, ext_col=(95, 58, 12), shadow=0.8, glow=((255, 190, 80), 0.3))
R.text_draw(out, "ЖОЛЫ", 560, 590, 210, fname="Black", grad=((255, 242, 196), (212, 148, 38)), extrude=12, ext_col=(95, 58, 12), shadow=0.8, glow=((255, 190, 80), 0.3))
cv2.line(out, (330, 735), (790, 735), (1.0, 0.82, 0.45), 3, cv2.LINE_AA)
R.text_draw(out, "1-ТОМ", 560, 810, 84, fname="Black", track=10, fill=(255, 255, 255), shadow=0.8)
cov = R.framed("cover.png", 330)
R.warp(out, cov, 1680, 790, ry=-22, rz=6, op=1.0)
out *= R.vignette()
Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(sys.argv[1])
