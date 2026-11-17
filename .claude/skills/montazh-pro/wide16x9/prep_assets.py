import numpy as np, cv2, os
from PIL import Image
D = os.path.dirname(os.path.abspath(__file__)); A = D + "/assets"
# 1) book cover (already RGBA, opaque)
im = Image.open(D + "/photos/IMG_1308.PNG").convert("RGBA"); im.save(A + "/cover.png")
# 2) 3D book: remove white background + reflection
im = np.array(Image.open(D + "/photos/IMG_1309.PNG").convert("RGB")).astype(np.int32)
h, w = im.shape[:2]
white = (im.min(axis=2) > 236).astype(np.uint8)
ff = white.copy(); mask = np.zeros((h + 2, w + 2), np.uint8)
for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
    cv2.floodFill(ff, mask, seed, 2)
bg = (ff == 2)
alpha = np.where(bg, 0, 255).astype(np.uint8)
# cut reflection: rows below the book bottom
rows = np.where((alpha > 0).sum(axis=1) > 0)[0]
col = (alpha[:, w // 2] > 0)
# find the book bottom: first gap in alpha rows after main body
prof = (alpha > 0).sum(axis=1)
body_end = None
for y in range(int(h * 0.6), h):
    if prof[y] < 5:
        body_end = y; break
if body_end: alpha[body_end:] = 0
alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
rgba = np.dstack([im.astype(np.uint8), alpha])
ys, xs = np.where(alpha > 10)
rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
Image.fromarray(rgba).save(A + "/book3d.png"); print("book3d", rgba.shape, body_end)
# 3) dombra: trim + downscale
d = Image.open(A + "/dombra_src.png").convert("RGBA"); d = d.crop(d.getbbox())
d = d.resize((int(d.width * 900 / d.height), 900), Image.LANCZOS); d.save(A + "/dombra.png"); print("dombra", d.size)
# 4) Abai photo: crop to Abai, segment person
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
M = "/home/user/claude-montazh-phone/.claude/skills/montazh-pro/models/selfie_multiclass.tflite"
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=M), output_confidence_masks=True))
src = np.array(Image.open(A + "/abai_src.jpg").convert("RGB"))
H, W = src.shape[:2]
crop = src[int(0.0 * H):int(0.62 * H), int(0.06 * W):int(0.80 * W)].copy()
r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
a = 1 - r.confidence_masks[0].numpy_view().astype(np.float32)
a = cv2.resize(a, (crop.shape[1], crop.shape[0]))
a = np.clip((a - 0.35) / 0.3, 0, 1)
a = cv2.GaussianBlur(a, (5, 5), 0)
gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
sep = np.dstack([gray * 1.0, gray * 0.93, gray * 0.80]).clip(0, 255).astype(np.uint8)  # sepia tint
out = np.dstack([sep, (a * 255).astype(np.uint8)])
Image.fromarray(out).save(A + "/abai.png"); print("abai", out.shape)
