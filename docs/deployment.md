# 民族服饰识别与AI绘图系统 — 部署流程操作文档

## 运行环境要求

### 最低配置

| 项目 | 要求 | 备注 |
|------|------|------|
| 操作系统 | Windows 10+ / Linux / macOS | 本项目在 Windows 10 上开发和测试 |
| Python | 3.10 | 3.11 及以上与 torch 2.4 有兼容问题 |
| GPU 显存 | **训练**: 6GB+ / **推理**: 3GB+ | CPU 也可运行但速度较慢 |
| CUDA | 12.4 | 可通过 PyTorch 官网安装对应版本 |
| 磁盘空间 | ~15GB | SD 模型 ~4GB + 依赖 ~3GB + 数据 ~2GB + 缓存 |
| 内存 | 8GB+ | |

### 本机参考配置（供对照）

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 10 21H2 (19045) |
| Python | 3.10 |
| GPU | NVIDIA RTX 2060 (6GB 显存) |
| CUDA | 12.4 |
| 包管理 | Conda |

---

## 步骤 1：创建 Conda 虚拟环境

```bash
conda create -n costume310 python=3.10 -y
conda activate costume310
```

---

## 步骤 2：安装 PyTorch (CUDA 12.4)

**方式一：在线安装（推荐）**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**方式二：手动下载 whl 安装（网络不稳定时使用）**

| 文件 | 链接 | 大小 |
|------|------|------|
| PyTorch 2.4.0 | `https://download.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp310-cp310-win_amd64.whl` | 2.4 GB |
| TorchVision 0.19.0 | `https://download.pytorch.org/whl/cu124/torchvision-0.19.0%2Bcu124-cp310-cp310-win_amd64.whl` | 5.6 MB |

```bash
# Linux/macOS 需替换为对应平台的 whl 文件名
pip install torch-2.4.0+cu124-cp310-cp310-win_amd64.whl
pip install torchvision-0.19.0+cu124-cp310-cp310-win_amd64.whl
```

**验证 CUDA 可用**：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 预期输出: True / <你的 GPU 名称>
```

---

## 步骤 3：解决 DLL 依赖（Windows 特有）

PyTorch 2.4.0 依赖两个 OpenMP 运行时库：

| DLL 文件 | 来源 | 解决方式 |
|----------|------|----------|
| `vcomp140.dll` | Microsoft Visual C++ Redistributable | 安装 [VC++ Redist](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `libomp140.x86_64.dll` | LLVM OpenMP Runtime | 放入 `torch/lib/` 目录 |

> 项目 `libs/` 目录下有 `libomp140.x86_64.dll` 备份，复制到 conda 环境的 `Lib/site-packages/torch/lib/` 下即可。

---

## 步骤 4：安装项目依赖

```bash
pip install -r requirements.txt
```

**注意**：安装 Gradio 后可能会出现版本冲突（huggingface-hub 被升级到 1.x），需要降级：

```bash
pip install "huggingface-hub<1.0,>=0.23.2"
pip install "numpy<2"
```

**关键依赖版本对照表**：

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
| safetensors | >=0.3.0 | LoRA 权重保存/加载 |

---

## 步骤 5：下载 Stable Diffusion 1.5 基础模型

由于 HuggingFace 在国内不可达，从 ModelScope（魔搭社区）下载。

**方式一：Git LFS 克隆（推荐）**

```bash
git clone https://www.modelscope.cn/AI-ModelScope/stable-diffusion-v1-5.git sd
```

**方式二：浏览器逐个下载**

访问 https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/files ，下载以下文件：

```
sd/
├── model_index.json
├── feature_extractor/
│   └── preprocessor_config.json
├── scheduler/
│   └── scheduler_config.json
├── tokenizer/
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── special_tokens_map.json
├── text_encoder/
│   ├── config.json
│   └── model.safetensors           (469 MB)
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors  (3.2 GB)
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors  (320 MB)
```

> `safety_checker/` 可跳过，训练和推理不需要。

**验证模型完整性**：

```bash
python scripts/test_model_load.py
# 预期输出: 全部模型组件加载成功!
```

---

## 步骤 6：准备 EasyNegative 文本反转嵌入（可选）

将 `EasyNegativeV2_V2.0.safetensors` 放入 `models/embeddings/` 目录。

此嵌入会在推理时自动加载，作为负向提示词前缀，改善生成质量。

**下载链接**：https://civitai.com/models/7808/easynegative （需注册）

---

## 步骤 7：数据采集与预处理

### 7.1 自动采集

```bash
python run_data.py --target 200 --source bing
```

### 7.2 手动补充

若采集数量不足，手动下载图片放入 `dataset/raw/<民族>/` 目录：

```
dataset/raw/
├── zang/    # 藏族服饰图片
├── qiang/   # 羌族服饰图片
└── yi/      # 彝族服饰图片
```

### 7.3 数据清洗

自动执行：去重（pHash）、去模糊（拉普拉斯检测）、统一尺寸（分类 224×224 + LoRA 512×512）、划分（70/15/15）。

### 7.4 验证数据集

```bash
# 检查各划分集数量
python -c "
from src.data_collection.dataset_utils import get_dataset_stats
for split in ['train', 'val', 'test']:
    s = get_dataset_stats(f'dataset/split/{split}')
    print(f'{split}: {s}')
"
```

---

## 步骤 8：训练分类模型

```bash
# 完整训练（两阶段）
python -m src.classification.train \
    --data dataset/split \
    --model models/classifier.pth \
    --batch 32 \
    --epochs1 10 \
    --epochs2 20
```

**训练流程**：
1. 阶段1：冻结 ResNet-50 主干，只训练分类头（lr=1e-3, 10 epochs）
2. 阶段2：解冻全网络微调（lr=1e-4, 20 epochs）
3. 自动保存最佳模型到 `models/classifier.pth`

### 8.1 模型评估

```bash
python -m src.classification.evaluate \
    --preds outputs/logs/test_predictions.pt \
    --model models/classifier.pth \
    --out outputs/figures
```

**预期指标**：测试准确率 ≥ 85%。

**本项目实际结果**：

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 藏族 | 0.88 | 0.92 | 0.90 |
| 羌族 | 0.92 | 0.85 | 0.88 |
| 彝族 | 0.86 | 0.88 | 0.87 |
| **总体** | **0.88** | **0.88** | **0.88** |

---

## 步骤 9：准备 LoRA 训练数据

```bash
python -m src.lora_training.prepare_dataset \
    --src dataset/cleaned \
    --out dataset/lora_data
```

此脚本会：
- 将图片居中裁剪为 512×512
- 为每张图片自动生成 `.txt` 描述文件（触发词 + 随机关键词）
- 输出到 `dataset/lora_data/<民族>/images/` 下

---

## 步骤 10：训练 LoRA 模型

### 10.1 配置确认

检查 `src/lora_training/config.py`：

```python
BASE_MODEL_ID = "<项目根目录>/sd"   # 本地 SD 1.5 路径，按实际修改
# ...
"max_train_steps": 500,    # 训练步数（150-200张图，500步足够）
"lora_rank": 16,
"lora_alpha": 32,
```

### 10.2 执行训练

```bash
# 训练全部三个民族（依次执行）
python -m src.lora_training.train_lora --ethnicity all --steps 500

# 或单独训练某个民族
python -m src.lora_training.train_lora --ethnicity zang --steps 500
```

**训练参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| max_train_steps | 500 | 150 张图 × 3 epochs ≈ 500 步 |
| learning_rate | 1e-4 | 余弦退火调度 |
| lora_rank | 16 | LoRA 低秩维度 |
| lora_alpha | 32 | LoRA 缩放因子 |
| batch_size | 1 | 低显存约束（可调大） |
| gradient_accumulation_steps | 4 | 等效 batch_size=4 |

**预期结果**：

| 民族 | 数据量 | 训练步数 | 最终 Loss | 输出文件 |
|------|--------|----------|-----------|----------|
| 藏族 | 167 张 | 500 | ~0.04 | `zang_lora.safetensors` (23MB) |
| 羌族 | 170 张 | 500 | ~0.006 | `qiang_lora.safetensors` (23MB) |
| 彝族 | 160 张 | 500 | ~0.017 | `yi_lora.safetensors` (23MB) |

训练完成后删除大 checkpoint 文件（`*_checkpoint_*.pt`，每个约 3.5GB），只保留 `.safetensors`。

---

## 步骤 11：启动 Web 演示系统

```bash
# 确保已激活 conda 环境
conda activate costume310

python run_web.py --port 8080
```

> `run_web.py` 内置环境检测——若当前 Python 非 3.10 或无 CUDA，会自动查找并切换到 `costume310` conda 环境重新启动。

访问 `http://localhost:8080`，包含三个 Tab：

| Tab | 功能 | 说明 |
|-----|------|------|
| 服饰识别 | 上传图片 → ResNet-50 分类 | 返回前三类别和置信度 |
| AI 绘图 | 选择民族 + 提示词 → SD+LoRA 生成 | 支持 8 种采样器、自定义正/反向提示词，首次加载约 10s，之后每张约 40s |
| 系统信息 | 技术栈和项目结构 | 静态展示 |

### 11.1 单独启动 FastAPI 后端

```bash
python run_web.py --api-only --port 8000
# API 文档: http://localhost:8000/docs
```

---

## 步骤 12：验证全系统

### 12.1 健康检查

```bash
curl http://localhost:8000/health
```

### 12.2 分类接口测试

```bash
curl -X POST http://localhost:8000/classify -F "file=@test_image.jpg"
```

### 12.3 文生图接口测试

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"ethnicity":"zang","prompt":"站在高原上","steps":20,"lora_scale":0.8}'
```

---

## 常见故障速查

| 症状 | 原因 | 解决 |
|------|------|------|
| `fbgemm.dll` 加载失败 | 缺少 LLVM/VC++ 运行时 | 见步骤 3 |
| `No module named 'torchvision'` | 安装 Gradio 时被卸载 | `pip install torchvision-...whl` |
| torchvision `no attribute 'extension'` | 在 base 环境 Python 3.13 运行 | 用 `conda activate costume310` 激活环境后运行 |
| `huggingface-hub>=0.23.2,<1.0 required` | 版本冲突 | `pip install "huggingface-hub<1.0,>=0.23.2"` |
| UNet `expected 4 channels, got 3` | 训练未用 VAE 编码 | 检查 `train_lora.py` 是否正确加载 VAE |
| `RuntimeError: CUDA out of memory` | 显存不足 | 降低分辨率或使用 CPU |
| Gradio "Method not implemented" | 后端函数报错 | 检查服务器日志定位具体异常 |
| ModelScope 下载超时 | 网络不稳 | 用浏览器下载或使用 git clone |
