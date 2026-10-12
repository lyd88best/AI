"""
step4_build_train_weather.py — Offline "Scheme B" mixed-weather TRAIN set.

For robustness/enhancement training we degrade the TRAIN split with random
adverse weather, keeping the dataset size unchanged (each image gets exactly
ONE randomly chosen state), so any gain is attributable to seeing diverse
degradations, not to more images. Fixed seed => fully reproducible.

Per training image, uniformly pick ONE of:
    sunny        -> copy clean image unchanged
    fog   (l1~l3 random) -> depth-driven scattering fog   (needs depth)
    rain  (l1~l3 random) -> rain streaks + depth veiling  (needs depth)
    lowlight (l1~l3 random) -> inverse-ISP low light       (no depth)

Labels are copied unchanged (pixel-only degradation; boxes unaffected).

Reuses the exact synthesis functions from step2/step2b/step3 so the train-time
and test-time degradations are identical in formulation.

Usage:
  python step4_build_train_weather.py \
      --img-dir dataset_split/images/train \
      --lbl-dir dataset_split/labels/train \
      --depth   dataset_split/depth_train \
      --out-img dataset_split/images/train_weather \
      --out-lbl dataset_split/labels/train_weather \
      --seed 0
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

# reuse synthesis logic (same formulation as the test-set weather)
import step2_synth_fog as fog
import step2b_synth_rain as rain
import step3_synth_lowlight as low

IMG_EXTS = (".jpg", ".jpeg", ".png")
FOG_LVLS = ["fog_l1", "fog_l2", "fog_l3"]
RAIN_LVLS = ["rain_l1", "rain_l2", "rain_l3"]
LOW_LVLS = ["lowlight_l1", "lowlight_l2", "lowlight_l3"]
STATES = ["sunny", "fog", "rain", "lowlight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--lbl-dir", required=True)
    ap.add_argument("--depth", required=True, help="depth dir for train imgs")
    ap.add_argument("--out-img", required=True)
    ap.add_argument("--out-lbl", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    img_dir, lbl_dir, depth_dir = Path(a.img_dir), Path(a.lbl_dir), Path(a.depth)
    out_img, out_lbl = Path(a.out_img), Path(a.out_lbl)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print(f"[ERROR] no images under {img_dir}")
        return

    # Deterministic per-image choice: fixed master rng drives all randomness.
    rng = np.random.default_rng(a.seed)
    # keep step modules' internal rng deterministic too (rain/lowlight use them)
    rain.rng = np.random.default_rng(a.seed)
    low.rng = np.random.default_rng(a.seed)

    counts = {s: 0 for s in STATES}
    missing_depth = 0
    manifest = []  # (name, state, level) for reproducibility log

    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[SKIP] cannot read {ip}")
            continue
        state = STATES[int(rng.integers(0, len(STATES)))]
        lbl = lbl_dir / (ip.stem + ".txt")
        level = "-"

        if state == "sunny":
            out = img
        elif state == "lowlight":
            level = LOW_LVLS[int(rng.integers(0, 3))]
            out = low.degrade_lowlight(img, **low.LEVELS[level])
        else:  # fog or rain need depth
            depth = fog.load_depth(depth_dir / (ip.stem + ".png"), img.shape)
            if depth is None:
                missing_depth += 1
                print(f"[SKIP->sunny] no depth for {ip.stem}")
                state, out = "sunny", img
            elif state == "fog":
                level = FOG_LVLS[int(rng.integers(0, 3))]
                out = fog.add_fog(img, depth, **fog.LEVELS[level])
            else:  # rain
                level = RAIN_LVLS[int(rng.integers(0, 3))]
                out = rain.add_rain(img, depth, **rain.LEVELS[level])

        cv2.imwrite(str(out_img / ip.name), out)
        dst = out_lbl / (ip.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, dst)
        else:
            dst.write_text("")
        counts[state] += 1
        manifest.append(f"{ip.name}\t{state}\t{level}")

    (out_img.parent.parent / "train_weather_manifest.tsv").write_text(
        "name\tstate\tlevel\n" + "\n".join(manifest), encoding="utf-8")

    total = sum(counts.values())
    print(f"[DONE] {total} mixed-weather train images -> {out_img}")
    print(f"       distribution: {counts}")
    if missing_depth:
        print(f"[WARN] {missing_depth} imgs had no depth -> fell back to sunny")
    print(f"       manifest: dataset_split/train_weather_manifest.tsv")


if __name__ == "__main__":
    main()
