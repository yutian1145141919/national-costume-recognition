# 民族服饰识别与AI绘图系统 — 使用帮助

## 目录

1. [环境配置详解](#1-环境配置详解)
2. [模块使用说明](#2-模块使用说明)
3. [API 接口文档](#3-api-接口文档)
4. [提示词编写指南](#4-提示词编写指南)
5. [常见问题与故障排除](#5-常见问题与故障排除)
6. [项目报告撰写建议](#6-项目报告撰写建议)

---

## 1. 环境配置详解

### 1.1 Python 环境

推荐使用 Python 3.10，高版本可能遇到依赖兼容性问题。

```bash
# 方法一：conda（推荐）
conda create -n costume310 python=3.10 -y
conda activate costume310

# 方法二：venv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 1.2 PyTorch 安装（CUDA 12.4）

本项目使用 PyTorch 2.4.0 + CUDA 12.4。

**在线安装（推荐）**：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**手动下载安装**（网络不稳定时，文件约 2.4GB）：

| 文件 | 链接 | 大小 |
|------|------|------|
| PyTorch 2.4.0 | `https://download.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp310-cp310-win_amd64.whl` | 2.4 GB |
| TorchVision 0.19.0 | `https://download.pytorch.org/whl/cu124/torchvision-0.19.0%2Bcu124-cp310-cp310-win_amd64.whl` | 5.6 MB |

```bash
pip install torch-2.4.0+cu124-cp310-cp310-win_amd64.whl
pip install torchvision-0.19.0+cu124-cp310-cp310-win_amd64.whl
```

### 1.3 DLL 依赖（Windows 特有）

PyTorch 2.4.0 依赖两个运行时库。如遇 `fbgemm.dll` 加载失败：

| DLL 文件 | 解决方式 |
|----------|----------|
| `vcomp140.dll` | 安装 [VC++ Redist](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `libomp140.x86_64.dll` | 复制 `libs/libomp140.x86_64.dll` 到 conda 环境的 `Lib/site-packages/torch/lib/` |

### 1.4 Stable Diffusion 基础模型

文生图功能使用本地 Stable Diffusion 1.5 模型 + LoRA 权重，通过 diffusers 库直接推理，**不依赖 SD WebUI**。

从 ModelScope（魔搭社区）下载模型文件到项目 `sd/` 目录：

```bash
git clone https://www.modelscope.cn/AI-ModelScope/stable-diffusion-v1-5.git sd
```

或从 [ModelScope 文件页](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/files) 手动下载，目录结构见 `docs/deployment.md` 步骤 5。

### 1.5 项目依赖安装

```bash
pip install -r requirements.txt
```

**关键版本兼容性**：

| 包 | 版本 | 备注 |
|----|------|------|
| torch | 2.4.0+cu124 | 必须匹配 CUDA 12.4 |
| torchvision | 0.19.0+cu124 | 必须匹配 torch |
| diffusers | 0.31.0 | 过高版本与 torch 2.4 不兼容 |
| peft | 0.19.1 | LoRA 训练和推理 |
| transformers | 4.44.2 | 过高版本 API 变更 |
| gradio | 6.x | Web 前端 |
| numpy | <2.0 | torch 2.4 要求 |
| huggingface-hub | 0.36.x | transformers 4.44 要求 <1.0 |

### 1.6 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 预期: True / <你的 GPU 名称>
```

---

## 2. 模块使用说明

### 2.1 数据采集

```bash
# 基础用法：每类采集200张
python -m src.data_collection.scraper --target 200 --source bing

# 或使用一键脚本
python run_data.py --target 200 --source bing
```

**手动补充数据**：

1. 手动下载图片放入 `dataset/raw/<民族名>/` 目录
2. 使用 Kaggle 等平台的开放数据集

### 2.2 数据预处理

预处理自动执行：去重（pHash）→ 去模糊（拉普拉斯检测）→ 统一尺寸（分类 224×224 + LoRA 512×512）→ 划分（70/15/15）。

```bash
# 一键清洗+划分
python run_data.py --target 200
```

**预处理参数**：

| 步骤 | 算法/参数 | 说明 |
|------|-----------|------|
| 去重 | pHash, 汉明距离 < 5 | 感知哈希去重 |
| 模糊检测 | 拉普拉斯方差 < 100 | 过滤低质量图片 |
| 分类尺寸 | 224×224 | ResNet-50 输入 |
| LoRA 尺寸 | 512×512 | SD 1.5 输入 |
| 划分比例 | 70/15/15 | train/val/test |

### 2.3 分类模型训练

```bash
# 使用一键脚本（推荐）
python run_train.py --type classifier --batch 32 --epochs1 10 --epochs2 20

# 或直接调用模块
python -m src.classification.train --data dataset/split --model models/classifier.pth --batch 32 --epochs1 10 --epochs2 20
```

**训练策略**：

| 阶段 | 说明 | 学习率 | 默认 epoch |
|------|------|--------|------------|
| 阶段1 | 冻结 ResNet-50 主干，只训练分类头 | 1e-3 | 10 |
| 阶段2 | 解冻全网络微调 | 1e-4 | 20 |

自动保存验证准确率最高的模型到 `models/classifier.pth`。

### 2.4 模型评估

```bash
python -m src.classification.evaluate \
  --preds outputs/logs/test_predictions.pt \
  --model models/classifier.pth \
  --out outputs/figures
```

**输出内容**：

| 文件 | 说明 |
|------|------|
| `confusion_matrix.png` | 混淆矩阵热力图 |
| `training_curves.png` | 训练 Loss 和 Accuracy 曲线 |
| `evaluation_report.txt` | 文字版评估报告 |

**本项目实际结果**（测试集准确率 88%）：

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 藏族 | 0.88 | 0.92 | 0.90 |
| 羌族 | 0.92 | 0.85 | 0.88 |
| 彝族 | 0.86 | 0.88 | 0.87 |

### 2.5 单张推理

```bash
python -m src.classification.inference "path/to/image.jpg" --top 3
```

### 2.6 LoRA 训练

```bash
# 准备训练数据
python -m src.lora_training.prepare_dataset --src dataset/cleaned --out dataset/lora_data

# 训练全部三个民族（各500步）
python -m src.lora_training.train_lora --ethnicity all --steps 500

# 或单独训练某个民族
python -m src.lora_training.train_lora --ethnicity zang --steps 500
```

**训练参数与配置**（`src/lora_training/config.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| max_train_steps | 500 | 150张图 × 3 epochs |
| learning_rate | 1e-4 | 余弦退火调度 |
| lora_rank | 16 | 低秩维度 |
| lora_alpha | 32 | 缩放因子 |
| batch_size | 1 | 6GB 显存约束 |
| gradient_accumulation_steps | 4 | 等效 batch_size=4 |

**本项目实际结果**：

| 民族 | 数据量 | 训练步数 | 最终 Loss | 输出文件 |
|------|--------|----------|-----------|----------|
| 藏族 | 167 张 | 500 | ~0.04 | `zang_lora.safetensors` (23MB) |
| 羌族 | 170 张 | 500 | ~0.006 | `qiang_lora.safetensors` (23MB) |
| 彝族 | 160 张 | 500 | ~0.017 | `yi_lora.safetensors` (23MB) |

**LoRA 训练故障排查**：

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss 不下降 | 学习率过低 | 增大 lr 至 2e-4 |
| 生成图过拟合 | 训练步数太多 | 减少 steps 至 300-400 |
| 风格不明显 | 数据量不足 | 增加训练图片至 50+ 张 |
| 人脸崩溃 | 学习率过高 | 降低 lr 至 5e-5 |
| `UNet expected 4 channels, got 3` | 训练未用 VAE 编码 | 检查 train_lora.py 是否正确加载 VAE |

### 2.7 Web 演示系统

```bash
# 启动完整 Gradio 界面（默认 http://localhost:8080）
python run_web.py --port 8080

# 或只启动 FastAPI 后端
python run_web.py --api-only --port 8000
```

界面包含三个 Tab：

| Tab | 功能 | 说明 |
|-----|------|------|
| 服饰识别 | 上传图片 → ResNet-50 分类 | 返回前三类别和置信度 |
| AI 绘图 | 选择民族 + 提示词 → SD+LoRA 生成 | 支持 8 种采样器、自定义正/反向提示词，首次加载约 10s，之后每张约 40s |
| 系统信息 | 技术栈和项目结构 | 静态展示 |

**AI 绘图参数说明**：
- **采样步数**：10~50 步，步数越多细节越好但速度越慢，推荐 25 步
- **采样器**：可选 DPM++ 2M Karras（默认，综合最佳）、Euler a（风格多样）、DDIM（经典）、UniPC 等 8 种
- **LoRA 权重**：0.1~1.5，推荐 0.6~0.9，越高越偏向目标风格
- **画面描述**：在正向提示词留空时生效，自动追加到预设提示词后
- **正向提示词**：可直接编写完整提示词，留空使用按民族的预设模板
- **负向提示词**：可直接编写排除内容，留空使用内置默认（EasyNegative + 通用质量负向词）
- **预设提示词已包含触发词**（如 `zang_costume`），自定义时也建议保留

---

## 3. API 接口文档

### 3.1 启动 API 服务

```bash
python run_web.py --api-only --port 8000
# API 文档: http://localhost:8000/docs
```

### 3.2 POST /classify — 图像识别

```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@test_image.jpg"
```

**响应**：

```json
{
  "label": "藏族",
  "confidence": 0.9234,
  "top3": [
    {"label": "藏族", "confidence": 0.9234, "index": 0},
    {"label": "羌族", "confidence": 0.0541, "index": 1},
    {"label": "彝族", "confidence": 0.0225, "index": 2}
  ]
}
```

### 3.3 POST /generate — 文生图

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "ethnicity": "zang",
    "prompt": "站在雪山前",
    "steps": 25,
    "lora_scale": 0.8,
    "cfg_scale": 7.0,
    "custom_positive": "",
    "custom_negative": ""
  }'
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| ethnicity | string | 是 | - | 民族: zang/qiang/yi |
| prompt | string | 否 | "" | 补充描述（custom_positive 为空时追加到预设后） |
| steps | int | 否 | 25 | 采样步数 10~50 |
| lora_scale | float | 否 | 0.8 | LoRA 权重 0.1~1.5 |
| cfg_scale | float | 否 | 7.0 | CFG 引导系数 |
| custom_positive | string | 否 | "" | 自定义正向提示词（覆盖预设） |
| custom_negative | string | 否 | "" | 自定义负向提示词（覆盖默认） |

**响应**：

```json
{
  "image_base64": "iVBORw0KGgo...",
  "positive_prompt": "zang_costume, ...",
  "negative_prompt": "easynegative, lowres, bad anatomy, ...",
  "parameters": {
    "ethnicity": "zang",
    "steps": 25,
    "cfg_scale": 7.0,
    "lora_scale": 0.8
  }
}
```

### 3.4 GET /health — 健康检查

```json
{
  "status": "ok",
  "classifier_available": true,
  "lora_models": ["zang_lora.safetensors", "qiang_lora.safetensors", "yi_lora.safetensors"]
}
```

### 3.5 GET /prompts — 提示词指南

返回三个民族的提示词撰写指南，包含触发词、风格关键词和示例提示词。

---

## 4. 提示词编写指南

### 4.1 藏族（zang_costume）

| 元素 | 关键词 |
|------|--------|
| 服装 | traditional Tibetan robe, colorful stripes, apron |
| 饰品 | silver jewelry, fur trim, turquoise |
| 场景 | plateau, snow mountain, grassland |

**示例**：

```
正向: zang_costume, traditional Tibetan robe, colorful stripes,
      silver ornaments, snow mountain background, masterpiece, best quality
负向: easynegative, lowres, bad anatomy, modern clothes, western style
```

### 4.2 羌族（qiang_costume）

| 元素 | 关键词 |
|------|--------|
| 服装 | traditional Qiang dress, embroidered patterns, sheepskin vest |
| 饰品 | silver headdress, cloud shoes, embroidered apron |
| 场景 | mountain village, stone tower, Sichuan valley |

### 4.3 彝族（yi_costume）

| 元素 | 关键词 |
|------|--------|
| 服装 | traditional Yi dress, pleated skirt, embroidered cape |
| 饰品 | silver ornaments, large earrings, headpiece |
| 场景 | Torch Festival, Yunnan mountains, Daliangshan |

### 4.4 提示词最佳实践

1. **触发词必须放在最前面**：`zang_costume, ...`
2. **LoRA 风格通过代码加载**，无需在提示词中写 `<lora:...>` 标签
3. **质量词放在最后**：`masterpiece, best quality, 8k`
4. **自定义正向提示词**：在 AI 绘图界面直接粘贴完整正向提示词即可覆盖预设；留空则使用内置模板
5. **自定义负向提示词**：可添加需要排除的元素（如 `modern clothes, wedding dress`），留空则使用内置 EasyNegative 嵌入 + 通用负向词
6. **LoRA 权重建议 0.6~0.9**：在界面上通过滑条调整，过低风格不明显，过高可能失真
7. **推理步数建议 20~30**：步数过少质量差，过多收益递减

---

## 5. 常见问题与故障排除

### 5.1 环境问题

**Q: `fbgemm.dll` 加载失败 / `ImportError: DLL load failed`**

缺少 OpenMP 运行时。解决：安装 VC++ Redist 并将 `libs/libomp140.x86_64.dll` 复制到 torch/lib 目录。详见 `docs/deployment.md` 步骤 3。

**Q: `No module named 'torchvision'`**

安装 Gradio 时可能卸载了 torchvision。重新安装与 torch 匹配的版本：

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Q: `partially initialized module 'torchvision' has no attribute 'extension'`**

在错误的 Python 环境中运行了项目（Python 版本不匹配或 torch/torchvision 版本不兼容）。常见于 Python 3.13 + torchvision 0.27.0+cpu 组合，存在循环导入 bug。

解决：切换到正确的 Python 3.10 环境后运行，或直接使用 `run_web.py`（已内置环境检测和自动切换）。

```bash
conda activate costume310
python run_web.py --port 8080
```

**Q: `huggingface-hub>=0.23.2,<1.0 required but found 1.x`**

Gradio 6.x 会升级 huggingface-hub 到 1.x，与 transformers 4.44 冲突：

```bash
pip install "huggingface-hub<1.0,>=0.23.2"
```

**Q: CUDA out of memory（显存不足）**

- 减小 LoRA 推理时使用 `pipe.enable_attention_slicing()` 和 `pipe.enable_vae_slicing()`（已默认启用）
- 分类训练减小 batch size：`--batch 8`

### 5.2 数据问题

**Q: 数据采集数量不足**

- 检查网络连接，切换来源 `--source baidu`
- 手动将图片放入 `dataset/raw/<民族>/` 目录
- 降低目标数量：`--target 100`

**Q: 划分后训练集太小无法训练**

确保每类至少有 20 张图片（建议 50+）。不足时使用数据增强扩充。

### 5.3 训练问题

**Q: 训练准确率不达标（< 85%）**

1. 检查数据集质量（是否有错误分类的图片）
2. 增加训练数据量
3. 增加 epoch 数
4. 调整学习率

**Q: 训练过程中 Loss 变成 NaN**

学习率过高导致梯度爆炸。降低学习率重试。

### 5.4 LoRA/推理问题

**Q: SD 基础模型从哪里下载？**

从 ModelScope（魔搭社区）下载，HuggingFace 在国内可能不可达：

```bash
git clone https://www.modelscope.cn/AI-ModelScope/stable-diffusion-v1-5.git sd
```

**Q: 生成图与目标风格不符**

- 调整 LoRA 权重（建议 0.6~0.9）
- 增加推理步数（建议 20~30）
- 检查提示词中的触发词是否正确

**Q: Gradio 界面 "Method not implemented" 错误**

通常是后端函数抛出异常。检查服务器终端日志定位具体错误。常见原因：模型文件缺失、torchvision 未安装、依赖版本冲突。

---

## 6. 项目报告撰写建议

项目报告建议包含以下章节（总计 ≥5000 字）：

### 6.1 方法说明（约 1500 字）

- 数据采集方法与来源
- 数据预处理流程（去重算法、质量过滤标准）
- ResNet-50 迁移学习原理
- LoRA（低秩适应）技术原理
- 两阶段训练策略的设计思路

### 6.2 实验结果（约 2000 字）

- 数据集统计信息（各类别数量、划分比例）
- 训练过程（Loss 曲线、Accuracy 曲线）
- 测试集评估结果（混淆矩阵、分类报告）
- 各类别精确率/召回率/F1-Score 分析
- LoRA 生成效果展示与对比

### 6.3 误差分析（约 1000 字）

- 各类别识别难易度分析
- 混淆矩阵中误分类案例的原因推断
- 数据不平衡对模型的影响
- 生成图中存在的问题与原因分析

### 6.4 改进建议（约 500 字）

- 数据层面：增加样本多样性、扩充数据集规模
- 模型层面：尝试 EfficientNet/ViT 等架构、超参数优化
- 工程层面：模型量化部署、移动端适配
- 应用层面：扩展到更多民族、服饰部件检测

---

## 参考资料

- [ResNet 论文](https://arxiv.org/abs/1512.03385)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion](https://stability.ai/)
- [diffusers 文档](https://huggingface.co/docs/diffusers/)
- [peft 文档](https://huggingface.co/docs/peft/)
- [ModelScope](https://www.modelscope.cn/)
