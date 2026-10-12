"""
step3_synth_lowlight.py — Low-light synthesis via inverse-ISP + physical noise.

Academic basis:
  - Inverse ISP (sRGB -> pseudo-RAW): Brooks et al., "Unprocessing Images for
    Learned Raw Denoising", CVPR 2019.
  - Physical noise (shot + read = Poisson-Gaussian): Chen et al. "Learning to
    See in the Dark" (SID, CVPR 2018); Wei et al. "Physics-based Noise Modeling"
    (ELD, CVPR 2020).
  (Verify exact titles/years before citing.)

GPU: NOT required. Pure CPU (numpy/opencv). Runs anywhere, incl. no-GPU host.

Pipeline per image:
  sRGB -> inverse gamma -> inverse white balance (approx) -> pseudo-linear RAW
       -> scale down exposure (ratio) -> add Poisson(shot)+Gaussian(read) noise
       -> forward ISP (white balance, gamma) -> low-light sRGB

Labels are copied unchanged (pixel-only degradation; boxes unaffected).

Usage (run on remote, no-GPU mode is fine):
  python step3_synth_lowlight.py --src <clean_dir> --out <out_dir>
  # <clean_dir> must contain images/ and labels/
  # multi-level output: images/lowlight_l1, lowlight_l2, lowlight_l3 (light->heavy)
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

rng = np.random.default_rng(0)  # fixed seed for reproducibility
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Three darkness levels.
#   ratio    : exposure scale (smaller = darker weak signal before noise)
#   shot/read: Poisson/Gaussian noise strength added on the weak signal
#   brighten : post-capture gain applied AFTER noise (amplifies noise, SID/ELD).
# Final relative brightness ~= ratio * brighten (kept < 1 so it stays dark),
# while the added noise is boosted by `brighten`. Heavier level = darker + noisier.
LEVELS = {
    "lowlight_l1": dict(ratio=0.30, shot=0.010, read=0.010, brighten=2.0),  # dusk / light
    "lowlight_l2": dict(ratio=0.15, shot=0.020, read=0.015, brighten=3.0),  # night / medium
    "lowlight_l3": dict(ratio=0.07, shot=0.035, read=0.025, brighten=4.0),  # extreme low light
}

# Approximate inverse/forward white-balance gains (R,G,B). Daylight-ish.
WB_GAINS = np.array([1.9, 1.0, 1.6], dtype=np.float32)  # RGB


def srgb_to_linear(img):
    """Inverse sRGB gamma: display sRGB [0,1] -> linear light."""
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)


def linear_to_srgb(img):
    """Forward sRGB gamma: linear light -> display sRGB [0,1]."""
    a = 0.055
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, img * 12.92, (1 + a) * (img ** (1 / 2.4)) - a)


def unprocess(img_bgr):
    """sRGB BGR uint8 -> pseudo-linear RAW (float RGB, [0,1]-ish)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)                 # undo gamma
    lin = lin / WB_GAINS[None, None, :]       # undo white balance
    return np.clip(lin, 0.0, 1.0)


def reprocess(lin_rgb):
    """pseudo-linear RAW -> sRGB BGR uint8."""
    lin = lin_rgb * WB_GAINS[None, None, :]   # apply white balance
    srgb = linear_to_srgb(lin)                # apply gamma
    out = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def add_poisson_gaussian(lin, shot, read):
    """Signal-dependent shot noise (Poisson approx) + read noise (Gaussian)."""
    var = shot * np.maximum(lin, 0.0) + read ** 2
    return lin + rng.normal(0.0, 1.0, lin.shape).astype(np.float32) * np.sqrt(var)


def degrade_lowlight(img_bgr, ratio, shot, read, brighten):
    lin = unprocess(img_bgr)                  # to pseudo RAW
    lin = lin * ratio                         # low exposure -> weak signal
    lin = add_poisson_gaussian(lin, shot, read)  # physical noise on the weak signal
    # Apply a brightening gain (as a camera/post-process would). The gain also
    # amplifies the noise added above, reproducing the "amplified noise after
    # gaining up a dark frame" behavior of SID (CVPR 2018) / ELD (CVPR 2020).
    # brighten < 1/ratio keeps the frame visibly dark while the noise is boosted.
    lin = lin * brighten
    lin = np.clip(lin, 0.0, 1.0)
    return reprocess(lin)                     # back to sRGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="clean dir with images/ and labels/")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS.keys()),
                    choices=list(LEVELS.keys()))
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    img_dir, lbl_dir = src / "images", src / "labels"
    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print(f"[ERROR] no images under {img_dir}")
        return

    for lv in a.levels:
        (out / "images" / lv).mkdir(parents=True, exist_ok=True)
        (out / "labels" / lv).mkdir(parents=True, exist_ok=True)

    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[SKIP] cannot read {ip}")
            continue
        lbl = lbl_dir / (ip.stem + ".txt")
        for lv in a.levels:
            deg = degrade_lowlight(img, **LEVELS[lv])
            cv2.imwrite(str(out / "images" / lv / ip.name), deg)
            dst = out / "labels" / lv / (ip.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, dst)
            else:
                dst.write_text("")
    print(f"[DONE] {len(imgs)} imgs x {len(a.levels)} levels -> {out}")


if __name__ == "__main__":
    main()
