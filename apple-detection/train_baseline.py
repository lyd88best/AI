"""
train_baseline.py —— 训练 baseline（原版 YOLOv8l + COCO 预训练，在你的苹果数据上微调）

公平对比要点（已内置）：
  - 从 config.PRETRAINED 加载同一预训练权重（魔改版也用它，别改）
  - 固定 SEED，锁死 EPOCHS/IMGSZ/BATCH（魔改版训练时必须用同样的值）
  - deterministic=True 尽量可复现

用法：
  pip install ultralytics
  python train_baseline.py
产出：
  runs/baseline/weights/best.pt   ← 这就是你的 baseline 模型
"""
from ultralytics import YOLO
import config as C


def main():
    # 加载 COCO 预训练权重作为迁移学习起点（不是终点！在你的数据上微调）
    model = YOLO(C.PRETRAINED)

    model.train(
        data=C.DATA_YAML,
        epochs=C.EPOCHS,
        imgsz=C.IMGSZ,
        batch=C.BATCH,
        seed=C.SEED,
        deterministic=True,        # 尽量可复现
        device=C.DEVICE,
        workers=C.WORKERS,
        cache=C.CACHE,             # RAM 缓存数据，消除 IO 瓶颈
        amp=C.AMP,                 # 混合精度，4090 提速省显存
        patience=C.PATIENCE,       # 早停
        project=C.RUNS_DIR,
        name="baseline",
        exist_ok=True,
        # 数据增强保持默认；若"恶劣天气增强"是你的贡献点，再单独开实验对比
    )
    print("\n[完成] baseline 权重: runs/baseline/weights/best.pt")
    print("下一步：python evaluate.py  以及  python failure_analysis.py")


if __name__ == "__main__":
    main()
