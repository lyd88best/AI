"""
step1_estimate_depth.py — Monocular depth estimation (shared input for fog & rain).

Academic basis:
  - Depth Anything V2 (2024), SOTA monocular relative-depth model.
  - Depth is required by the atmospheric scattering model used for fog synthesis
    (Foggy Cityscapes paradigm; Sakaridis et al., IJCV 2018) and by the rain
    veiling layer. (Verify exact titles/years before citing.)

GPU: REQUIRED. Run on the remote machine in GPU ("有卡") mode.
     This is a ONE-TIME step; its output (depth maps) is reused by fog and rain,
     so you only pay for GPU once.

Output:
  <out>/<image_stem>.png   (16-bit depth map, near=small, far=large, normalized)

Depends:
  pip install torch opencv-python numpy
  pip install depth_anything_v2   (or clone the official repo and add to PYTHONPATH)

Model weights (download once; use a mirror if GitHub is slow):
  Depth-Anything-V2-Small/Base/Large .pth

Usage (remote, GPU mode):
  python step1_estimate_depth.py --src <clean_dir> --out <depth_dir> \
      --encoder vitb --weights /path/to/depth_anything_v2_vitb.pth
  # <clean_dir> must contain images/
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_model(encoder: str, weights: str, device: str):
    """Load Depth Anything V2. Keeps import local so the file is importable
    even on a no-GPU host where the package may be absent."""
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    cfgs = {
        "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
        "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
        "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
    }
    model = DepthAnythingV2(**cfgs[encoder])
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="clean dir containing images/")
    ap.add_argument("--out", required=True, help="output dir for depth maps")
    ap.add_argument("--encoder", default="vitb", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--weights", required=True, help="path to Depth Anything V2 .pth")
    ap.add_argument("--input-size", type=int, default=518)
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] CUDA not available — running on CPU will be slow. "
              "This step is meant for GPU mode.")

    model = load_model(a.encoder, a.weights, device)

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (src / "images").glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print(f"[ERROR] no images under {src/'images'}")
        return

    for i, ip in enumerate(imgs, 1):
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[SKIP] cannot read {ip}")
            continue
        # DA-V2 relative model outputs DISPARITY (near=large, far=small).
        disp = model.infer_image(img, a.input_size).astype(np.float32)
        # We need a per-image relative near->far gradient to drive fog density,
        # NOT a metric depth. We tried the physically-exact reciprocal
        # (depth ~ 1/disparity) but on monocular disparity of near-field-dominated
        # orchard scenes it collapses all foreground to ~0 (sky disparity ~0 blows
        # up and flattens the rest), yielding an unusable depth. A linear inversion
        # of the normalized disparity preserves the full foreground gradient and
        # gives a clean near-dark / far-bright map (verified on test images), which
        # is what the scattering model needs (far=large -> lower transmission).
        d = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
        d_far = 1.0 - d                               # near->0, far->1
        d16 = (d_far * 65535.0).astype(np.uint16)
        cv2.imwrite(str(out / (ip.stem + ".png")), d16)
        if i % 50 == 0:
            print(f"[{i}/{len(imgs)}] done")

    print(f"[DONE] depth maps for {len(imgs)} images -> {out}")
    print("Note: depth stored as 16-bit PNG, normalized per-image, far=large.")


if __name__ == "__main__":
    main()
