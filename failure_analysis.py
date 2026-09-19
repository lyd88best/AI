"""
baseline 失败分析：按天气条件分别评估 YOLOv8，找出 baseline 在雾/雨/低光/晴天/小目标上各差多少。

前提数据结构（按天气分文件夹）：
  dataset/
    images/
      test/
        sunny/     *.jpg
        rainy/     *.jpg
        foggy/     *.jpg
        lowlight/  *.jpg
    labels/
      test/
        sunny/     *.txt   (YOLO 格式，和 images 同名)
        rainy/     *.txt
        foggy/     *.txt
        lowlight/  *.txt

用法：
  pip install ultralytics
  python failure_analysis.py
"""

from pathlib import Path
import yaml
from ultralytics import YOLO

# ============ 改这里 ============
WEIGHTS = "runs/detect/train/weights/best.pt"   # 你训练好的 baseline 权重
DATASET_ROOT = Path("dataset")                   # 数据集根目录
WEATHERS = ["sunny", "rainy", "foggy", "lowlight"]
CLASS_NAMES = ["apple"]                          # 你的类别；分成熟/未熟就写 ["ripe","unripe"]
IMGSZ = 640
DEVICE = 0                                        # GPU 编号；CPU 用 "cpu"
# ================================

def make_yaml(weather: str) -> str:
    """为单个天气子集生成临时 data.yaml（val 指向该子集，Ultralytics 用 val 划分评估）。"""
    cfg = {
        "path": str(DATASET_ROOT.resolve()),
        "val": f"images/test/{weather}",   # 评估这个子集
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    out = DATASET_ROOT / f"_eval_{weather}.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)

def main():
    model = YOLO(WEIGHTS)
    rows = []
    for w in WEATHERS:
        img_dir = DATASET_ROOT / "images" / "test" / w
        if not img_dir.exists():
            print(f"[跳过] {img_dir} 不存在")
            continue
        data_yaml = make_yaml(w)
        # split="val" 因为我们把该天气子集配到了 val 字段
        m = model.val(data=data_yaml, imgsz=IMGSZ, device=DEVICE, split="val", verbose=False)
        b = m.box
        rows.append({
            "weather": w,
            "mAP50":       round(float(b.map50), 4),   # AP50
            "mAP50-95":    round(float(b.map),   4),   # AP（主指标）
            "precision":   round(float(b.mp),    4),
            "recall":      round(float(b.mr),    4),
        })

    # ---- 打印失败分析表 ----
    print("\n================ Baseline 分天气失败分析 ================")
    print(f"{'天气':<10}{'mAP50':>10}{'mAP50-95':>12}{'Precision':>12}{'Recall':>10}")
    print("-" * 54)
    for r in rows:
        print(f"{r['weather']:<10}{r['mAP50']:>10}{r['mAP50-95']:>12}{r['precision']:>12}{r['recall']:>10}")

    if rows:
        # 以 sunny 为基准，算每种恶劣天气掉了多少点
        base = next((r for r in rows if r["weather"] == "sunny"), rows[0])
        print("\n---- 相对晴天(sunny)的 mAP50-95 掉点 ----")
        for r in rows:
            if r["weather"] == base["weather"]:
                continue
            drop = round(base["mAP50-95"] - r["mAP50-95"], 4)
            print(f"{r['weather']:<10} 掉点 {drop:+.4f}  ({'重灾区' if drop > 0.05 else '轻微'})")
        # 找出最该优先攻的痛点
        worst = min((r for r in rows), key=lambda x: x["mAP50-95"])
        print(f"\n>>> 最该优先攻的痛点：{worst['weather']}（mAP50-95 最低 = {worst['mAP50-95']}）")

if __name__ == "__main__":
    main()
