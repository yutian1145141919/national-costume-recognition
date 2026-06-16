# 民族服饰识别与AI绘图综合实践项目

## 项目简介

本项目以少数民族非物质文化遗产保护为背景，融合计算机视觉、深度学习与生成式 AI 等前沿技术，围绕**藏族、羌族、彝族**三个民族的传统服饰图像，构建一套完整的「数据采集 → 模型训练 → 智能识别 → AI 绘图」闭环系统。

### 核心功能

| 功能模块 | 说明 |
|----------|------|
| **数据采集** | 自动从公开图片源搜索并下载民族服饰图片 |
| **数据预处理** | 感知哈希去重、模糊检测、尺寸统一、数据增强 |
| **图像识别** | 基于 ResNet-50 的三分类模型，测试准确率 88% |
| **LoRA 风格模型** | 基于 Stable Diffusion 1.5 的民族服饰风格微调 |
| **Web 演示** | Gradio 三Tab界面：上传识别 + 文生图 + 系统信息 |

### 适用场景

- 数据科学与大数据技术专业综合实践/课程设计
- 民族文化数字化保护项目
- 计算机视觉与 AIGC 教学案例

## 环境要求

### 最低配置

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 10+ / Linux / macOS |
| Python | 3.10（3.11+ 有兼容问题） |
| GPU 显存 | 3GB+（仅推理）/ 6GB+（含训练） |
| 内存 | 8GB+ |
| 磁盘 | 15GB+ |
| CUDA | 12.4（或 11.8/12.1，与 PyTorch 版本匹配） |

### 本机参考配置

| 组件 | 配置 |
|------|------|
| 操作系统 | Windows 10 21H2 |
| GPU | NVIDIA RTX 2060 (6GB) |
| CUDA | 12.4 |
| Python | 3.10 (conda: costume310) |

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n costume310 python=3.10 -y
conda activate costume310

# 安装 PyTorch（CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
pip install -r requirements.txt
```

> 详细环境配置（含 DLL 依赖处理、SD 模型下载）见 [docs/deployment.md](docs/deployment.md)

### 2. 数据采集与预处理

```bash
# 一键执行：爬虫下载 → 清洗 → 划分数据集
python run_data.py --target 200
```

### 3. 模型训练

```bash
# 训练分类模型（ResNet-50，目标准确率 ≥85%）
python run_train.py --type classifier --batch 32

# 训练 LoRA 风格模型（需要 GPU 和本地 SD 1.5 模型）
python -m src.lora_training.prepare_dataset --src dataset/cleaned --out dataset/lora_data
python -m src.lora_training.train_lora --ethnicity all --steps 500
```

### 4. 启动 Web 演示

```bash
# 确保已激活 conda 环境
conda activate costume310
python run_web.py --port 8080
```

> `run_web.py` 内置环境检测——若当前 Python 非 3.10 或无 CUDA，会自动查找并切换到 costume310 环境。

访问 `http://localhost:8080`，包含服饰识别、AI 绘图、系统信息三个 Tab。

### 5. 一键全流程

```bash
python run.py
```

## 项目结构

```
民族lora/
├── CLAUDE.md                   # 项目协作规范
├── README.md                   # 项目说明（本文件）
├── requirements.txt            # Python 依赖清单
├── run.py                      # 一键全流程入口
├── run_data.py                 # 数据采集+预处理入口
├── run_train.py                # 模型训练入口
├── run_web.py                  # Web 服务入口
├── docs/                       # 项目文档
│   ├── deployment.md           # 部署流程操作文档
│   ├── HELP.md                 # 详细使用帮助
│   └── specification.md        # 技术规格文档
├── libs/                       # 运行时依赖库
│   └── libomp140.x86_64.dll   # LLVM OpenMP 运行时
├── cache/                      # 缓存文件
├── sd/                         # Stable Diffusion 1.5 模型文件
├── dataset/                    # 数据集目录
│   ├── raw/                    # 原始图片（zang/qiang/yi）
│   ├── cleaned/                # 清洗后图片
│   ├── split/                  # 数据集划分（train/val/test）
│   ├── lora_data/              # LoRA 训练数据
│   └── annotations.csv         # 标注记录
├── models/                     # 模型存储
│   ├── classifier.pth          # 分类模型权重
│   ├── lora/                   # LoRA 权重文件（.safetensors）
│   └── embeddings/             # 文本反转嵌入
│       └── EasyNegativeV2.safetensors
├── src/                        # 源代码
│   ├── data_collection/        # 数据采集模块
│   │   ├── scraper.py          # 网络图片爬虫
│   │   └── dataset_utils.py    # 数据集工具
│   ├── data_preprocessing/     # 数据预处理模块
│   │   ├── clean.py            # 数据清洗
│   │   ├── split.py            # 数据集划分
│   │   └── augment.py          # 数据增强
│   ├── classification/         # 图像分类模块
│   │   ├── train.py            # 模型训练
│   │   ├── evaluate.py         # 模型评估
│   │   └── inference.py        # 推理接口
│   ├── lora_training/          # LoRA 训练模块
│   │   ├── config.py           # 训练配置
│   │   ├── prepare_dataset.py  # 数据准备
│   │   └── train_lora.py       # LoRA 训练
│   └── web/                    # Web 演示模块
│       ├── api.py              # FastAPI 后端
│       ├── app.py              # Gradio 前端
│       ├── inference.py        # 本地 SD+LoRA 推理
│       └── prompts.py          # 提示词模板
├── outputs/                    # 输出结果
│   ├── figures/                # 评估图表
│   ├── logs/                   # 训练日志
│   └── samples/                # 生成样图
└── scripts/                    # 工具脚本
    ├── test_model_load.py      # SD 模型完整性验证
    ├── test_embedding.py       # EasyNegative 嵌入测试
    ├── test_inference.py       # SD+LoRA 推理测试
    └── download_sd15.py        # SD 模型下载脚本
```

## 技术架构

```
┌─────────────────────────────────────────────────┐
│                   展示层                          │
│         Gradio 前端 / FastAPI 接口                │
├─────────────────────────────────────────────────┤
│                   服务层                          │
│    /classify（识别）    /generate（文生图）        │
├─────────────────────────────────────────────────┤
│                   模型层                          │
│  ResNet-50（分类）  +  SD 1.5 + LoRA（生成）      │
├─────────────────────────────────────────────────┤
│                   数据层                          │
│  原始图像 → 标注 → 清洗 → 划分（训练/验证/测试）    │
└─────────────────────────────────────────────────┘
```

## 交付成果

| 成果 | 要求 | 状态 |
|------|------|------|
| 数据集 | 三类服饰各 ≥200 张，含标注文件 | `dataset/annotations.csv` |
| 识别模型 | 测试集准确率 ≥85%，含混淆矩阵与分类报告 | 88%，见 `outputs/figures/` |
| LoRA 权重 | 可在 Stable Diffusion 加载并生效 | 3个民族，见 `models/lora/` |
| Web 系统 | 支持图像上传识别 + 文本输入生图 | `python run_web.py` |
| 项目文档 | README + HELP + 部署文档 + 规格文档 | 见 `docs/` |

### 分类模型实测结果

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 藏族 | 0.88 | 0.92 | 0.90 |
| 羌族 | 0.92 | 0.85 | 0.88 |
| 彝族 | 0.86 | 0.88 | 0.87 |
| **总体** | **0.88** | **0.88** | **0.88** |

### LoRA 训练实测结果

| 民族 | 数据量 | 训练步数 | 最终 Loss | 输出文件 |
|------|--------|----------|-----------|----------|
| 藏族 | 167 张 | 500 | ~0.04 | `zang_lora.safetensors` |
| 羌族 | 170 张 | 500 | ~0.006 | `qiang_lora.safetensors` |
| 彝族 | 160 张 | 500 | ~0.017 | `yi_lora.safetensors` |

## 常见问题

更多问题请参阅 [docs/HELP.md](docs/HELP.md) 和 [docs/deployment.md](docs/deployment.md)

### Q: 数据采集失败怎么办？

部分图片来源可能网络不可达。可以手动下载图片放入 `dataset/raw/<民族名>/` 目录，或使用 Kaggle 等开放数据集。

### Q: 训练时显存不足？

- 分类训练：`python run_train.py --batch 8`
- LoRA 推理已默认启用 attention slicing 和 VAE slicing

### Q: SD 模型从哪里下载？

从 ModelScope（魔搭社区）下载，HuggingFace 在国内可能不可达：

```bash
git clone https://www.modelscope.cn/AI-ModelScope/stable-diffusion-v1-5.git sd
```

### Q: fbgemm.dll 加载失败？

缺少 OpenMP 运行时。安装 VC++ Redist 并将 `libs/libomp140.x86_64.dll` 复制到 conda 环境的 `Lib/site-packages/torch/lib/`。

### Q: torchvision 循环导入报错 "no attribute 'extension'"？

在错误的 Python 环境中运行了项目。激活 costume310 后运行：

```bash
conda activate costume310
python run_web.py --port 8080
```
