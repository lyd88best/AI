"""
evaluate.py —— 统一评估：对任意 YOLO 权重输出 mAP(全套) + Params + Latency + FPS

所有模型（baseline / 魔改版 / 其他 SOTA）都走这一个评估流程，数字才严格可比
（这正是教授那张对比表整齐可比的原因）。

用法：
  python evaluate.py                      # 默认评估 baseline
  python evaluate.py runs/mymod/weights/best.pt  MyMod   # 评估指定权重并命名
可作为模块被 build_table.py 调用。
"""
import sys
import time
import torch
from ultralytics import YOLO
import config as C


def count_params(model) -> float:
    """返回参数量（百万 M）。"""
    m = model.model
    n = sum(p.numel() for p in m.parameters())
    return round(n / 1e6, 2)


def measure_fps(weights: str):
    """在固定输入尺寸下测单张延迟(ms)和 FPS。"""
    model = YOLO(weights)
    dev = "cuda:0" if (C.DEVICE != "cpu" and torch.cuda.is_available()) else "cpu"
    dummy = torch.zeros(1, 3, C.IMGSZ, C.IMGSZ)
    # 预热
    for _ in range(C.FPS_WARMUP):
        model.predict(dummy, imgsz=C.IMGSZ, device=dev, verbose=False)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(C.FPS_RUNS):
        model.predict(dummy, imgsz=C.IMGSZ, device=dev, verbose=False)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / C.FPS_RUNS
    return round(dt * 1000, 2), round(1.0 / dt, 1)   # (Lat ms, FPS)


def evaluate(weights: str, name: str = "model", split: str = "test") -> dict:
    """跑一个模型，返回一行指标 dict。"""
    model = YOLO(weights)
    m = model.val(
        data=C.DATA_YAML, imgsz=C.IMGSZ, batch=C.BATCH,
        conf=C.CONF, iou=C.IOU_NMS, device=C.DEVICE,
        split=split, verbose=False,
    )
    b = m.box
    params = count_params(model)
    lat, fps = measure_fps(weights)
    row = {
        "model": name,
        "Params(M)": params,
        "mAP50":    round(float(b.map50), 4),
        "mAP50-95": round(float(b.map),   4),   # 主指标 AP
        "mAP75":    round(float(b.map75), 4),
        "precision": round(float(b.mp),   4),
        "recall":    round(float(b.mr),   4),
        "Lat(ms)":  lat,
        "FPS":      fps,
    }
    return row


def _print_row(row: dict):
    print("\n================ 评估结果 ================")
    for k, v in row.items():
        print(f"  {k:<12}: {v}")


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else "runs/baseline/weights/best.pt"
    name = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    _print_row(evaluate(weights, name))
