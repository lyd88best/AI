"""
step2b_synth_rain.py — Rain synthesis: physical rain streaks + depth-based veiling.

Academic basis:
  - Rain photometric/geometric appearance: Garg & Nayar, "Vision and Rain"
    (IJCV 2007) and related.
  - Rain accumulation / veiling effect modeled as depth-based haze (reuses the
    scattering model and the depth from step1), consistent with how modern
    rain-synthesis works combine streaks + atmospheric veiling.
  (Verify exact titles/years before citing.)

GPU: NOT required. Pure CPU. Reuses depth maps from step1.

Pipeline per image (per rain level):
  1. atmosphere: darken + desaturate + depth-based veiling (uses depth)
  2. streaks: multi-layer oriented motion-blurred streaks (near longer/brighter)
  3. composite streaks over the atmospheric background
  4. slight global blur (rain curtain)

Labels copied unchanged (pixel-only; boxes unaffected).

Usage (remote, no-GPU mode is fine):
  python step2b_synth_rain.py --src <clean_dir> --depth <depth_dir> --out <out_dir>
  # <clean_dir>: images/ + labels/ ; <depth_dir>: 16-bit depth PNGs from step1
  # multi-level output: images/rain_l1, rain_l2, rain_l3 (light->heavy)
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

rng = np.random.default_rng(0)  # fixed seed for reproducibility
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Three rain intensities.
#   drops: streak count; length/thickness: streak geometry;
#   darkness/saturation: atmosphere; veil: depth-based rain-haze strength.
LEVELS = {
    "rain_l1": dict(drops=1500, length=18, thickness=1, darkness=0.80,
                    saturation=0.85, veil=0.08, A=0.80),   # light rain
    "rain_l2": dict(drops=2800, length=22, thickness=1, darkness=0.68,
                    saturation=0.72, veil=0.14, A=0.82),   # medium rain
    "rain_l3": dict(drops=4200, length=28, thickness=2, darkness=0.58,
                    saturation=0.62, veil=0.22, A=0.84),   # heavy rain
}
MAX_DEPTH_M = 3.0


def load_depth(depth_path: Path, shape):
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / 65535.0
    if d.shape[:2] != shape[:2]:
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return d


def apply_atmosphere(img_bgr, depth, darkness, saturation, veil, A):
    """Darken + desaturate + depth-based veiling (rain accumulation haze)."""
    out = img_bgr.astype(np.float32)
    out *= darkness
    gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = out * saturation + gray3 * (1 - saturation)
    # depth-based veiling: farther regions get more haze (like fog, lighter)
    t = np.exp(-veil * (depth * MAX_DEPTH_M))[..., None]
    A255 = A * 255.0
    out = out * t + A255 * (1 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_streak_layer(h, w, drops, length, thickness, angle):
    """One oriented, motion-blurred rain streak layer."""
    layer = np.zeros((h, w), dtype=np.uint8)
    dx = int(length * np.cos(np.deg2rad(angle)))
    dy = int(length * np.sin(np.deg2rad(angle)))
    for _ in range(drops):
        x = int(rng.integers(0, w)); y = int(rng.integers(0, h))
        val = int(rng.integers(120, 190))
        cv2.line(layer, (x, y), (x + dx, y + dy), val, thickness)
    # motion blur along the streak direction
    k = max(3, length // 3)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    layer = cv2.filter2D(layer, -1, kernel)
    return layer


def add_rain(img_bgr, depth, drops, length, thickness, darkness,
             saturation, veil, A):
    h, w = img_bgr.shape[:2]
    bg = apply_atmosphere(img_bgr, depth, darkness, saturation, veil, A)
    # two layers with slightly different angles -> more natural
    l1 = make_streak_layer(h, w, drops, length, thickness, angle=-65)
    l2 = make_streak_layer(h, w, drops // 2, int(length * 1.3), thickness, angle=-60)
    streaks = cv2.add(l1, l2)
    streaks = cv2.cvtColor(streaks, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(bg, 0.85, streaks, 0.28, 0)
    out = cv2.blur(out, (3, 3))  # rain curtain softening
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="clean dir with images/ and labels/")
    ap.add_argument("--depth", required=True, help="depth dir from step1")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS.keys()),
                    choices=list(LEVELS.keys()))
    a = ap.parse_args()

    src, depth_dir, out = Path(a.src), Path(a.depth), Path(a.out)
    img_dir, lbl_dir = src / "images", src / "labels"
    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print(f"[ERROR] no images under {img_dir}")
        return

    for lv in a.levels:
        (out / "images" / lv).mkdir(parents=True, exist_ok=True)
        (out / "labels" / lv).mkdir(parents=True, exist_ok=True)

    missing = 0
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[SKIP] cannot read {ip}")
            continue
        depth = load_depth(depth_dir / (ip.stem + ".png"), img.shape)
        if depth is None:
            missing += 1
            print(f"[SKIP] no depth for {ip.stem} (run step1 first)")
            continue
        lbl = lbl_dir / (ip.stem + ".txt")
        for lv in a.levels:
            deg = add_rain(img, depth, **LEVELS[lv])
            cv2.imwrite(str(out / "images" / lv / ip.name), deg)
            dst = out / "labels" / lv / (ip.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, dst)
            else:
                dst.write_text("")
    print(f"[DONE] rain for {len(imgs)-missing} imgs x {len(a.levels)} levels -> {out}")
    if missing:
        print(f"[WARN] {missing} images skipped (no depth map).")


if __name__ == "__main__":
    main()
