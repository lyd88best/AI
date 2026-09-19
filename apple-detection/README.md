# 苹果恶劣天气检测 —— baseline + 统一评估脚手架

围绕"公平对比 + 论文可直接用"设计。全部超参集中在 `config.py`，改一处即可。

## 目录
```
apple-detection/
├── config.py              # 全局配置（超参、路径、公平对比锁死的变量）
├── configs/apple.yaml     # 数据集定义 + 目录结构说明
├── train_baseline.py      # 训 baseline（yolov8l.pt 预训练微调，固定种子）
├── evaluate.py            # 统一评估：mAP全套 + Params + Lat + FPS
├── failure_analysis.py    # 分天气失败分析（找痛点，指导选亮点）
├── build_table.py         # 一键生成论文对比表（Markdown + CSV）
├── requirements.txt
└── (运行后生成) runs/  results/
```

## 快速开始
```bash
pip install -r requirements.txt

# 1. 按 configs/apple.yaml 底部说明摆好数据（图片 + YOLO txt 标注）
#    test 集按天气分文件夹：test/{sunny,rainy,foggy,lowlight}/

# 2. 训 baseline（得到 runs/baseline/weights/best.pt）
python train_baseline.py

# 3. 统一评估 baseline（mAP + Params + FPS）
python evaluate.py

# 4. 分天气失败分析（看雾/雨/低光各差多少 → 决定加什么模块）
python failure_analysis.py

# 5. 训练魔改版后，在 build_table.py 的 MODELS 里登记权重，一键出对比表
python build_table.py
```

## 公平对比铁律（脚手架已内置，别破坏）
- baseline 和所有魔改版用**同一个** `config.PRETRAINED`（yolov8l.pt）
- 同一数据集 + 同一 train/val/test 划分
- 同样的 EPOCHS / IMGSZ / BATCH / SEED
- **只有"模型结构"这一个变量允许不同**

## 加魔改版的流程（数据+baseline就绪后）
1. 复制一份 YOLOv8l 结构 yaml，插入你的模块（如 CBAM / CA / 增强前端），存成 `yolov8l-mymod.yaml`
2. 训练：`YOLO("yolov8l-mymod.yaml").load("yolov8l.pt")`（老层继承预训练，新层自动随机初始化）
   —— 训练超参与 baseline 完全一致
3. 在 `build_table.py` 的 `MODELS` 登记，跑 `build_table.py` 出对比表
4. 每个模块单独做消融，证明各自有效

## 注意
- 这套脚手架需要你的**真实数据**才能跑出真实数字。数据未就绪前，可先 `pip install` 并通读脚本。
- 权重/日志较大，`runs/` 建议加进 `.gitignore`。
