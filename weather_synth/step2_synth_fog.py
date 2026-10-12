"""
step2_synth_fog.py — Fog synthesis via atmospheric scattering + estimated depth.

Academic basis:
  - Atmospheric scattering model I = J*t + A*(1-t), t = exp(-beta*d)
    (Narasimhan & Nayar; He et al. Dark Channel Prior, CVPR 2009).
  - Depth-driven fog synthesis paradigm: Foggy Cityscapes
    (Sakaridis et al., IJCV 2018). Here depth comes from step1 (estimated),
    replacing sensor depth since MinneApple has none.
  (Verify exact titles/years before citing.)

GPU: NOT required. Pure CPU. Depth is precomputed by step1_estimate_depth.py.

Pipeline per image (per fog level beta):
  read J (clean) + d (depth, far=large, [0,1])
  t = exp(-beta * d_meters)        # d scaled to a pseudo metric range
  guided-filter t to respect edges
  I = J*t + A*(1-t)

Labels copied unchanged (pixel-only; boxes unaffected).

Usage (remote, no-GPU mode is fine):
  python step2_synth_fog.py --src <clean_dir> --depth <depth_dir> --out <out_dir>
  # <clean_dir>: images/ + labels/ ; <depth_dir>: 16-bit depth PNGs from step1
  # multi-level output: images/fog_l1, fog_l2, fog_l3 (light->heavy)
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")

# Three fog densities (beta). Larger beta = thicker fog.
# d is scaled to [0, MAX_DEPTH_M] meters before applying exp(-beta*d).
LEVELS = {
    "fog_l1": dict(beta=0.6, A=0.85),   # light haze
    "fog_l2": dict(beta=1.2, A=0.88),   # medium fog
    "fog_l3": dict(beta=2.2, A=0.90),   # heavy fog
}
MAX_DEPTH_M = 3.0  # pseudo metric scale for normalized depth -> controls falloff


def guided_filter(guide_gray, src, radius=30, eps=1e-3):
    """Edge-aware smoothing of transmission using the image as guide."""
    g = guide_gray.astype(np.float32)
    p = src.astype(np.float32)
    mean_g = cv2.boxFilter(g, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))
    corr_gp = cv2.boxFilter(g * p, -1, (radius, radius))
    corr_gg = cv2.boxFilter(g * g, -1, (radius, radius))
    cov_gp = corr_gp - mean_g * mean_p
    var_g = corr_gg - mean_g * mean_g
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    return mean_a * g + mean_b


def load_depth(depth_path: Path, shape):
    """Load 16-bit depth PNG, return float [0,1] (far=large), resized if needed."""
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / 65535.0
    if d.shape[:2] != shape[:2]:
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return d


def add_fog(img_bgr, depth, beta, A):
    h, w = img_bgr.shape[:2]
    J = img_bgr.astype(np.float32) / 255.0
    d_m = depth * MAX_DEPTH_M                      # pseudo meters
    t = np.exp(-beta * d_m)                        # transmission
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    t = guided_filter(gray, t)                     # edge-aware
    t = np.clip(t, 0.05, 1.0)[..., None]
    I = J * t + A * (1 - t)
    return (np.clip(I, 0, 1) * 255).astype(np.uint8)


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
            deg = add_fog(img, depth, **LEVELS[lv])
            cv2.imwrite(str(out / "images" / lv / ip.name), deg)
            dst = out / "labels" / lv / (ip.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, dst)
            else:
                dst.write_text("")
    print(f"[DONE] fog for {len(imgs)-missing} imgs x {len(a.levels)} levels -> {out}")
    if missing:
        print(f"[WARN] {missing} images skipped (no depth map).")


if __name__ == "__main__":
    main()
