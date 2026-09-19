"""
failure_analysis.py —— 分天气失败分析：找出 baseline 在雾/雨/低光/晴天上各差多少。

原理：把 test 集按天气分文件夹，每个子集单独评估，以晴天为基准算掉点，
      自动点名"最该优先攻的痛点" —— 这就是你选魔改亮点的数据依据。

前提：test 图按天气分文件夹（见 configs/apple.yaml 底部结构说明）：
  dataset/images/test/{sunny,rainy,foggy,lowlight}/*.jpg
  dataset/labels/test/{sunny,rainy,foggy,lowlight}/*.txt

用法：
  python failure_analysis.py [weights]
  weights 省略则用 runs/baseline/weights/best.pt
"""
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO
import config as C

DATASET_ROOT = Path(C.DATA_YAML).parent.parent / "dataset"  # 依据 apple.yaml 的 path 习惯


def make_yaml(weather: str) -> str:
    cfg = {
        "path": str(DATASET_ROOT.resolve()),
        "val": f"images/test/{weather}",   # 把该天气子集配到 val 字段来评估
        "names": {i: n for i, n in enumerate(C.CLASS_NAMES)},
    }
    out = DATASET_ROOT / f"_eval_{weather}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else "runs/baseline/weights/best.pt"
    model = YOLO(weights)
    rows = []
    for w in C.WEATHERS:
        img_dir = DATASET_ROOT / "images" / "test" / w
        if not img_dir.exists():
            print(f"[跳过] {img_dir} 不存在")
            continue
        m = model.val(data=make_yaml(w), imgsz=C.IMGSZ, conf=C.CONF, iou=C.IOU_NMS,
                      device=C.DEVICE, split="val", verbose=False)
        b = m.box
        rows.append({
            "weather": w,
            "mAP50": round(float(b.map50), 4),
            "mAP50-95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4),
            "recall": round(float(b.mr), 4),
        })

    if not rows:
        print("没有可评估的天气子集，请检查数据目录结构。")
        return

    print("\n================ Baseline 分天气失败分析 ================")
    print(f"{'天气':<10}{'mAP50':>10}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}")
    print("-" * 54)
    for r in rows:
        print(f"{r['weather']:<10}{r['mAP50']:>10}{r['mAP50-95']:>12}{r['precision']:>12}{r['recall']:>10}")

    base = next((r for r in rows if r["weather"] == "sunny"), rows[0])
    print(f"\n---- 相对 {base['weather']} 的 mAP50-95 掉点 ----")
    for r in rows:
        if r["weather"] == base["weather"]:
            continue
        drop = round(base["mAP50-95"] - r["mAP50-95"], 4)
        tag = "重灾区" if drop > 0.05 else "轻微"
        print(f"{r['weather']:<10} 掉点 {drop:+.4f}  ({tag})")

    worst = min(rows, key=lambda x: x["mAP50-95"])
    print(f"\n>>> 最该优先攻的痛点：{worst['weather']}（mAP50-95 最低 = {worst['mAP50-95']}）")

    # 存 CSV 供论文用
    out = Path(C.RESULTS_DIR); out.mkdir(exist_ok=True)
    import csv
    with (out / "failure_analysis.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print(f"\n已保存: {out / 'failure_analysis.csv'}")


if __name__ == "__main__":
    main()
