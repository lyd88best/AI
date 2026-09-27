"""
build_table.py —— 一键生成论文对比表：把多个模型统一评估，输出 Markdown + CSV 表。

在 MODELS 里登记你要对比的模型（baseline / 魔改版 / 其他 SOTA），
全部走 evaluate.py 的同一流程，产出教授那种"精度 + 参数 + 速度"一览表。

用法：
  python build_table.py
产出：
  results/comparison.md   results/comparison.csv
"""
import csv
from pathlib import Path
from evaluate import evaluate
import config as C

# ============ 在这里登记要对比的模型 ============
# (显示名, 权重路径)  —— 训好一个登记一个
MODELS = [
    ("YOLOv8l (baseline)", "runs/baseline/weights/best.pt"),
    # ("Ours (+CA)",            "runs/ca/weights/best.pt"),
    # ("Ours (+CA+Enhance)",    "runs/ca_enh/weights/best.pt"),
    # ("YOLOv9-C",              "runs/yolov9c/weights/best.pt"),
    # ("YOLO11l",               "runs/yolo11l/weights/best.pt"),
]
# ================================================

COLS = ["model", "imgsz", "maxDets", "Params(M)",
        "mAP50-95", "mAP50", "mAP75", "APs", "APm", "APl",
        "precision", "recall", "Lat(ms)", "FPS"]


def main():
    rows = []
    for name, weights in MODELS:
        if not Path(weights).exists():
            print(f"[跳过] 权重不存在: {weights}")
            continue
        print(f"评估中: {name} ...")
        rows.append(evaluate(weights, name))

    if not rows:
        print("没有可评估的模型，请先训练并在 MODELS 里登记权重路径。")
        return

    # 按主指标 mAP50-95 降序（和教授的表一致：组内按 AP 降序）
    rows.sort(key=lambda r: r["mAP50-95"], reverse=True)

    out = Path(C.RESULTS_DIR); out.mkdir(exist_ok=True)

    # ---- Markdown 表（可直接贴进论文/报告）----
    md = ["| " + " | ".join(COLS) + " |",
          "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in COLS) + " |")
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    # ---- CSV ----
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=COLS)
        wtr.writeheader()
        wtr.writerows(rows)

    print("\n================ 对比表 ================")
    print("\n".join(md))
    print(f"\n已保存: {out/'comparison.md'} 和 {out/'comparison.csv'}")


if __name__ == "__main__":
    main()
