# 恶劣天气合成使用说明（顶会级：雾 / 雨 / 低光）

基于 MinneApple（普通 RGB，无深度、无 RAW）合成雾、雨、低光三种恶劣天气测试集。
方法均对应学术界标准做法，不是手工叠加。

> **本机（Windows）无显卡**：所有需要 GPU 的操作都在远程 AutoDL 跑。
> 下表标明每一步要不要卡。

---

## 一、脚本总览与执行顺序

| 步骤 | 脚本 | 要显卡 | 说明 |
|------|------|--------|------|
| 1 | `step1_estimate_depth.py` | ✅ 要 | 估计深度，**一次性**，雾和雨共用其输出 |
| 2 | `step2_synth_fog.py` | ❌ 不要 | 雾合成（散射模型 + 步骤1的深度）|
| 2b | `step2b_synth_rain.py` | ❌ 不要 | 雨合成（雨条纹 + 深度雨雾）|
| 3 | `step3_synth_lowlight.py` | ❌ 不要 | 低光合成（逆ISP + 物理噪声），**不依赖深度** |

**依赖关系**：
```
step1 (深度, 有卡)
   ├──> step2  雾   (无卡)
   └──> step2b 雨   (无卡)
step3 低光 (无卡, 独立, 不需要 step1)
```

**最省算力的跑法**：
1. 远程切「有卡模式」→ 只跑 step1（算深度，几十分钟），算完存盘
2. 远程切回「无卡模式」→ 跑 step2 / step2b / step3（纯 CPU，可反复调参）
3. 低光（step3）不依赖深度，无卡模式随时可先跑

---

## 二、输入 / 输出目录约定

**输入**（干净数据，YOLO 格式）：
```
<clean_dir>/
  images/*.png      原图
  labels/*.txt      YOLO 标注（class cx cy w h，归一化）
```
以当前项目为例，`<clean_dir>` 用划分出的测试集扁平目录：
`dataset_split/test_clean`

**输出**（每种天气按强度分子目录，标注原样复制）：
```
<out_dir>/
  images/{fog_l1,fog_l2,fog_l3}/*.png
  labels/{fog_l1,fog_l2,fog_l3}/*.txt
  （雨同理 rain_l1/2/3，低光 lowlight_l1/2/3）
```
l1→l3 = 轻→重，三档强度（对标 Foggy Cityscapes 的多浓度做法）。
标注只做像素退化不改框，因此各强度标签内容相同、直接复制。

---

## 三、具体命令

> 远程激活环境（AutoDL）：
> ```bash
> source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
> ```

### 步骤 1：估计深度（有卡模式）

先装 Depth Anything V2 并下载权重（权重走国内镜像，避免 GitHub 慢）：
```bash
pip install opencv-python numpy torch
# 安装 depth_anything_v2（官方包或 clone 仓库加入 PYTHONPATH）
# 下载权重，如 depth_anything_v2_vitb.pth
```
运行：
```bash
python step1_estimate_depth.py \
  --src dataset_split/test_clean \
  --out dataset_split/depth \
  --encoder vitb \
  --weights /path/to/depth_anything_v2_vitb.pth
```
产出：`dataset_split/depth/<图名>.png`（16 位深度图，远=大）。

### 步骤 2：雾合成（无卡模式）
```bash
python step2_synth_fog.py \
  --src dataset_split/test_clean \
  --depth dataset_split/depth \
  --out dataset_split/weather_fog
```

### 步骤 2b：雨合成（无卡模式）
```bash
python step2b_synth_rain.py \
  --src dataset_split/test_clean \
  --depth dataset_split/depth \
  --out dataset_split/weather_rain
```

### 步骤 3：低光合成（无卡模式，不需要深度）
```bash
python step3_synth_lowlight.py \
  --src dataset_split/test_clean \
  --out dataset_split/weather_lowlight
```

---

## 四、调参位置（都在各脚本顶部的 LEVELS 字典）

| 天气 | 关键参数 | 作用 |
|------|---------|------|
| 雾 | `beta` | 雾浓度，越大越浓；`A` 大气光（天空色）|
| 雨 | `drops/length/thickness` 雨丝；`darkness/saturation` 氛围；`veil` 雨雾 |
| 低光 | `ratio` 曝光（越小越暗）；`shot/read` 物理噪声强度 |

改完直接重跑对应脚本即可。只改像素、不动标注。

---

## 五、方法出处（投稿前务必自行核实准确标题/年份/venue）

- **雾**：大气散射模型（Narasimhan & Nayar；He et al. DCP, CVPR 2009）；
  深度驱动合成范式 Foggy Cityscapes（Sakaridis et al., IJCV 2018）；
  深度来源 Depth Anything V2 (2024)。
- **雨**：雨外观物理模型（Garg & Nayar, "Vision and Rain", IJCV 2007）；
  雨累积/veiling 用基于深度的散射雾层建模。
- **低光**：逆 ISP（Brooks et al., "Unprocessing...", CVPR 2019）；
  物理噪声 Poisson-Gaussian（Chen et al. SID, CVPR 2018；Wei et al. ELD, CVPR 2020）。

> ⚠️ 以上文献的确切标题/作者/年份/venue 未经联网核实，引用前请在
> Google Scholar 逐一确认并阅读原文。

---

## 六、顶会级注意事项（务必做到）

1. **多强度档位**：每种天气出 l1/l2/l3 三档，别只做一档。
2. **标注不变**：三种都只改像素，YOLO 标注复用（脚本已自动处理）。
3. **可复现**：所有脚本固定随机种子，公开全部参数（LEVELS 字典）。
4. **真实数据验证（最关键）**：顶会不认纯合成评估。必须**额外在真实恶劣
   天气检测基准**上评估，例如 雾→RTTS、低光→ExDark、综合→ACDC/BDD100K，
   证明方法在真实世界有效。MinneApple 合成只能作为受控实验之一。
5. **合成真实性是必要非充分**：肉眼看着像 ≠ 顶会认可；关键是方法有出处、
   深度是估计的而非假深度。
6. **雨最薄弱**：学术界对合成雨真实性质疑最多，冲顶会建议雨额外引入真实雨
   数据或学习式（GAN/扩散）方法佐证。

---

## 七、验证合成效果

合成图是普通 PNG，可直接用看图软件打开查看效果。建议每种天气各抽查几张
（轻/中/重各一张），确认：
- 效果可见且自然（不过曝/不过暗/雨丝不过假）
- 目标（苹果）未被过度遮挡，仍可检测
- 深度驱动的雾/雨雾呈现"近清晰、远朦胧"的合理分布
