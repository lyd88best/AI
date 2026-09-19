# =====================================================================
# 全局实验配置 —— 所有脚本共享这里的设置，保证公平对比的"变量锁死"
# 改这一个文件即可，不要在各脚本里散落硬编码。
# =====================================================================

# ---- 预训练权重（baseline 和魔改版必须用同一个，公平对比铁律）----
PRETRAINED = "yolov8l.pt"          # COCO 预训练起点；换尺寸改这里（yolov8m.pt 等）
MODEL_YAML_BASELINE = "yolov8l.yaml"   # baseline 的结构定义（原版）

# ---- 数据集 ----
DATA_YAML = "configs/apple.yaml"   # 数据集定义（下方 apple.yaml）
CLASS_NAMES = ["apple"]            # 你的类别；分成熟度就写 ["ripe", "unripe"]
WEATHERS = ["sunny", "rainy", "foggy", "lowlight"]   # 失败分析的天气子集

# ---- 训练超参（baseline 和所有魔改版必须完全一致）----
EPOCHS = 150
IMGSZ = 640
BATCH = 16
SEED = 0                           # 固定随机种子，保证可复现
DEVICE = 0                         # GPU 编号；CPU 用 "cpu"
WORKERS = 8

# ---- 评估 / 测速 ----
CONF = 0.001                       # 评估用低置信度阈值（算 mAP 标准做法）
IOU_NMS = 0.7
FPS_WARMUP = 20                    # 测 FPS 前的预热次数
FPS_RUNS = 100                     # 测 FPS 的正式次数

# ---- 输出 ----
RUNS_DIR = "runs"
RESULTS_DIR = "results"            # 对比表、失败分析表输出到这
