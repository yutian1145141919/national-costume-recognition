# 民族服饰识别与AI绘图系统 — 技术规格文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 项目名称 | 民族服饰识别与AI绘图综合实践系统 |
| 版本 | 1.0.3 |
| 最后更新 | 2026-06-16 |
| 运行环境 | Windows / Linux / macOS, Python 3.10, CUDA 12.4 |

---

## 1. 系统概述

### 1.1 系统目标

构建一套面向藏族、羌族、彝族三类传统民族服饰的完整 AI 管线，覆盖数据采集、图像分类识别、AI 图像生成三个核心环节，并以 Web 应用形式对外提供交互服务。

### 1.2 核心能力

| 能力 | 描述 | 技术载体 |
|------|------|----------|
| 数据采集 | 自动从公开图源搜索并下载民族服饰图像 | requests + BeautifulSoup |
| 数据预处理 | 去重、质量过滤、尺寸统一、数据集划分 | OpenCV + imagehash |
| 服饰识别 | 输入任意图片，判断所属民族并给出置信度 | ResNet-50（迁移学习） |
| 风格生成 | 输入文本描述，生成指定民族风格的服饰图片 | Stable Diffusion 1.5 + LoRA |
| Web 服务 | 提供 HTTP API 和图形界面 | FastAPI + Gradio |

### 1.3 非功能需求

| 维度 | 要求 |
|------|------|
| 分类准确率 | 测试集 ≥ 85% |
| 推理响应时间 | 分类 < 2s / 生成 < 60s（GPU） |
| 生成分辨率 | 512×512（可扩展） |
| 显存占用 | LoRA 推理 ≤ 3GB |
| 并发支持 | 单用户（Gradio 会话模式） |
| 离线运行 | 全部模型本地加载，不依赖外网 API |

---

## 2. 系统架构

### 2.1 分层架构

```
┌──────────────────────────────────────────────────┐
│              展示层 (Presentation)                 │
│  ┌─────────────┐  ┌─────────────────────────────┐ │
│  │  Gradio UI  │  │  FastAPI REST (Swagger)     │ │
│  └──────┬──────┘  └──────────────┬──────────────┘ │
├─────────┼────────────────────────┼────────────────┤
│         │      服务层 (Service)   │                │
│  ┌──────┴────────────────────────┴──────────────┐ │
│  │  /classify（图像识别）  /generate（文生图）    │ │
│  │  /health（健康检查）    /prompts（提示词指南） │ │
│  └──────┬────────────────────────┬──────────────┘ │
├─────────┼────────────────────────┼────────────────┤
│         │      模型层 (Model)     │                │
│  ┌──────┴──────────┐  ┌──────────┴──────────────┐ │
│  │  ResNet-50      │  │  SD 1.5 + LoRA          │ │
│  │  (分类推理)      │  │  + EasyNegative TI      │ │
│  └─────────────────┘  └─────────────────────────┘ │
├──────────────────────────────────────────────────┤
│              数据层 (Data)                         │
│  ┌────────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ 原始图像    │  │ 清洗数据  │  │ 划分数据集     │ │
│  │ (raw/)     │  │ (cleaned/) │  │ (split/)      │ │
│  └────────────┘  └──────────┘  └───────────────┘ │
└──────────────────────────────────────────────────┘
```

### 2.2 数据流

```
[网络爬虫] → dataset/raw/ → [清洗] → dataset/cleaned/ → [划分] → dataset/split/
                                                                        │
                                                          ┌─────────────┤
                                                    train/ │      val/   │    test/
                                                       │   │       │    │      │
                                                  [训练] │   │       │    │      │
                                                       ▼   ▼       ▼    ▼      ▼
                                              models/classifier.pth   [评估]
                                                                        │
                                                                  outputs/figures/

[清洗后图片] → [LoRA数据准备] → dataset/lora_data/ → [LoRA训练] → models/lora/*.safetensors

[用户上传图片] → /classify → ResNet-50 → 分类结果 → JSON 响应
[用户输入文本] → /generate → SD 1.5 + LoRA → 图片 → Base64 响应
```

---

## 3. 技术栈

### 3.1 运行时环境

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10 | 运行语言（3.11+ 有兼容问题） |
| CUDA | 12.4（或 11.8/12.1） | GPU 加速，需与 PyTorch 版本匹配 |
| Conda | 任意版本 | 虚拟环境管理（可选，也可用 venv） |

### 3.2 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.4.0+cu124 | 深度学习框架 |
| torchvision | 0.19.0+cu124 | 视觉模型 + 预训练权重 |
| diffusers | 0.31.0 | Stable Diffusion 推理 |
| peft | 0.19.1 | LoRA 适配（低秩微调） |
| transformers | 4.44.2 | CLIP 文本编码器 |
| safetensors | ≥0.3.0 | LoRA 权重序列化 |
| accelerate | ≥1.5.0 | 分布式/混合精度训练 |

### 3.3 Web 与应用依赖

| 包 | 版本 | 用途 |
|----|------|------|
| gradio | 6.x | Web 图形界面 |
| fastapi | ≥0.110 | REST API 后端 |
| uvicorn | ≥0.30 | ASGI 服务器 |
| Pillow | ≥10.0 | 图像处理 |
| opencv-python | ≥4.9 | 计算机视觉（预处理） |
| scikit-learn | ≥1.5 | 评估指标 |
| matplotlib | ≥3.8 | 图表绘制 |
| seaborn | ≥0.13 | 混淆矩阵可视化 |
| pandas | ≥2.2 | 数据处理 |
| imagehash | ≥4.3 | 感知哈希去重 |

### 3.4 版本锁定原因

| 约束 | 原因 |
|------|------|
| numpy < 2.0 | torch 2.4 与 numpy 2.x ABI 不兼容 |
| huggingface-hub < 1.0 | transformers 4.44.2 要求 <1.0；Gradio 6.x 会升级到 1.x |
| diffusers ≤ 0.31.0 | 更高版本要求更新的 torch/transformers |
| transformers = 4.44.x | 高版本 API 变更（model_max_length 等） |

---

## 4. 模型规格

### 4.1 分类模型 — ResNet-50

| 属性 | 值 |
|------|-----|
| 架构 | ResNet-50（torchvision 实现） |
| 预训练权重 | ImageNet V1 |
| 输入尺寸 | 224×224×3（RGB） |
| 输出类别 | 3（藏族 / 羌族 / 彝族） |
| 模型大小 | ~97MB（.pth 格式） |
| 推理延迟 | < 0.5s/张（GPU）/ < 2s/张（CPU） |

**训练策略**：

| 阶段 | 可训练参数 | 学习率 | Epochs | 优化器 |
|------|-----------|--------|--------|--------|
| 阶段1（分类头） | fc 层 | 1e-3 | 10 | Adam |
| 阶段2（全网络） | 所有层 | 1e-4 | 20 | Adam (weight_decay=1e-4) |

**学习率调度**：ReduceLROnPlateau, mode='min', patience=2, factor=0.5  
**早停**：验证准确率连续 5 epoch 未提升则停止  
**数据划分**：70% 训练 / 15% 验证 / 15% 测试（分层抽样）

### 4.2 生成模型 — Stable Diffusion 1.5 + LoRA

| 属性 | 值 |
|------|-----|
| 基础模型 | Stable Diffusion 1.5（diffusers 格式） |
| 参数量 | ~980M（UNet ~860M + CLIP ~123M + VAE ~84M） |
| 推理精度 | FP16（GPU）/ FP32（CPU） |
| 采样器 | DPMSolverMultistepScheduler |
| 输入 | 文本提示词（≤77 tokens） |
| 输出 | 512×512 RGB 图片 |
| 推理时间 | ~40s/张（参考：RTX 2060 6GB, 25 steps），取决于 GPU 性能 |

**LoRA 配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| r (rank) | 16 | 低秩矩阵维度 |
| alpha | 32 | 缩放因子（实际 scale = alpha/r = 2.0） |
| target_modules | to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2 | 作用于注意力 + FF 层 |
| dropout | 0.05 | 防止过拟合 |
| 训练步数 | 500 | ~3 epochs（150 张图, batch=1, grad_accum=4） |
| 学习率 | 1e-4 | 余弦退火调度 |
| 权重大小 | ~23MB/民族 | safetensors 格式 |

**推理优化**（低显存适配）：

| 技术 | 说明 |
|------|------|
| attention_slicing | 将注意力计算分片，减少峰值显存 |
| VAE slicing | 将 VAE 解码分片为 512×512 块 |
| FP16 推理 | 半精度浮点，显存减半 |
| safety_checker=None | 跳过安全检查器，节省约 600MB |

**嵌入**：

| 文件 | 用途 |
|------|------|
| EasyNegativeV2.safetensors | 文本反转负向嵌入，改善生成质量 |

---

## 5. API 规格

### 5.1 端点总览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 系统健康检查 |
| POST | `/classify` | 服饰图像分类 |
| POST | `/generate` | AI 文生图 |
| GET | `/prompts` | 提示词撰写指南 |
| GET | `/docs` | Swagger UI（FastAPI 自动生成） |

### 5.2 GET /health

**响应模型**：

```json
{
  "status": "ok",
  "classifier_available": true,
  "classifier_path": "models/classifier.pth",
  "lora_models": ["zang_lora.safetensors", "qiang_lora.safetensors", "yi_lora.safetensors"],
  "embedding_available": true,
  "sd_model_path": "sd/",
  "device": "cuda"
}
```

### 5.3 POST /classify

**请求**：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 图片文件（JPG/PNG/BMP/WebP） |

**图片预处理**：Resize(256)→CenterCrop(224)→ToTensor→Normalize(ImageNet μ,σ)

**响应**：

```json
{
  "label": "藏族",
  "confidence": 0.9234,
  "top3": [
    {"index": 0, "label": "藏族", "confidence": 0.9234},
    {"index": 1, "label": "羌族", "confidence": 0.0541},
    {"index": 2, "label": "彝族", "confidence": 0.0225}
  ]
}
```

**状态码**：

| 状态码 | 说明 |
|--------|------|
| 200 | 识别成功 |
| 400 | 未提供图片文件 |
| 503 | 分类模型未加载 |

### 5.4 POST /generate

**请求**：`application/json`

| 字段 | 类型 | 必填 | 默认值 | 范围 | 说明 |
|------|------|------|--------|------|------|
| ethnicity | string | 是 | - | zang/qiang/yi | 目标民族 |
| prompt | string | 否 | "" | ≤500 字符 | 补充描述（custom_positive 为空时追加到预设后） |
| steps | int | 否 | 25 | 10~50 | 推理步数 |
| lora_scale | float | 否 | 0.8 | 0.1~1.5 | LoRA 权重 |
| cfg_scale | float | 否 | 7.0 | 1.0~15.0 | CFG 引导系数 |
| seed | int | 否 | -1 | -1 或 0~2³¹-1 | 随机种子 |
| custom_positive | string | 否 | "" | ≤500 字符 | 自定义正向提示词（覆盖预设模板） |
| custom_negative | string | 否 | "" | ≤500 字符 | 自定义负向提示词（覆盖默认 EasyNegative） |
| sampler | string | 否 | "DPM++ 2M Karras" | 见列表 | 采样器：DPM++ 2M Karras/DPM++ SDE Karras/Euler a/Euler/DDIM/UniPC/PNDM/LMS |

**响应**：

```json
{
  "image_base64": "iVBORw0KGgo...",
  "positive_prompt": "zang_costume, traditional Tibetan robe, ..., masterpiece, best quality",
  "negative_prompt": "easynegative, lowres, bad anatomy, blurry, deformed, ...",
  "parameters": {
    "ethnicity": "zang",
    "steps": 25,
    "cfg_scale": 7.0,
    "lora_scale": 0.8,
    "seed": 12345678
  }
}
```

**状态码**：

| 状态码 | 说明 |
|--------|------|
| 200 | 生成成功，返回 Base64 编码的 PNG |
| 400 | 参数无效（未知民族等） |
| 503 | 模型加载失败或显存不足 |

---

## 6. 数据集规格

### 6.1 数据采集

| 属性 | 值 |
|------|-----|
| 数据来源 | 必应图片搜索（Bing Image Search） |
| 目标数量 | 每类 ≥ 200 张 |
| 搜索关键词 | "藏族服饰" / "羌族服饰" / "彝族服饰" |
| 过滤条件 | JPG/PNG/BMP/WebP 格式, ≥ 10KB, 短边 ≥ 256px |
| 采集工具 | `src/data_collection/scraper.py` |

### 6.2 数据预处理

| 步骤 | 方法 | 参数 |
|------|------|------|
| 去重 | 感知哈希（pHash） | 汉明距离阈值 = 5 |
| 模糊检测 | 拉普拉斯方差 | 阈值 = 100 |
| 尺寸统一（分类） | Resize + CenterCrop | 224×224 |
| 尺寸统一（LoRA） | thumbnail + 黑边填充（保持宽高比） | 512×512 |

### 6.3 数据集划分

| 子集 | 比例 | 用途 |
|------|------|------|
| train | 70% | 模型训练 |
| val | 15% | 超参数调优、早停判断 |
| test | 15% | 最终评估 |

划分方式：分层抽样（stratified split），各类别比例在各子集中保持一致。

### 6.4 数据增强（训练时）

| 变换 | 参数 |
|------|------|
| RandomHorizontalFlip | p=0.5 |
| RandomRotation | ±15° |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 |
| RandomResizedCrop | scale=(0.8, 1.0) |

### 6.5 LoRA 训练数据

每张图片配对一个同名的 `.txt` 文件，内容格式：

```
<触发词>, <风格描述关键词>
```

触发词：`zang_costume` / `qiang_costume` / `yi_costume`

---

## 7. 系统安全

### 7.1 输入校验

| 接口 | 校验项 |
|------|--------|
| POST /classify | 文件类型白名单（JPG/PNG/BMP/WebP）、文件大小 ≤ 10MB |
| POST /generate | ethnicity 枚举校验、steps 范围 10~50、prompt/custom_positive/custom_negative ≤ 500 字符、lora_scale 范围 0.1~1.5 |

### 7.2 已知限制

- **单用户模式**：Gradio 为会话模式，不支持并发多用户
- **本地部署**：所有模型本地加载，无外部 API 依赖，也意味着不提供公网访问
- **显存限制**：LoRA 推理需要 ≥ 3GB 可用显存；同时运行分类和生成可能触发 OOM
- **无认证机制**：FastAPI 和 Gradio 均未添加认证层，仅适合本地或内网使用

---

## 8. 性能基准

### 8.1 最低硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU 显存 | 3GB（仅推理）/ 6GB（含训练） | 8GB+ |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 15GB | 20GB+ (SSD) |
| CPU | 4 核 | 8 核+ |

### 8.2 参考测试环境

| 组件 | 配置 |
|------|------|
| CPU | Intel Core i7-10750H |
| GPU | NVIDIA GeForce RTX 2060 (6GB) |
| RAM | 16GB DDR4 |
| OS | Windows 10 21H2 |

### 8.3 分类模型

| 指标 | 值 |
|------|-----|
| 测试准确率 | 88% |
| 推理时间（GPU） | 0.3s/张 |
| 推理时间（CPU） | 1.5s/张 |

### 8.4 生成模型

| 指标 | 值 |
|------|-----|
| 模型加载时间 | ~10s |
| 推理时间（25 steps） | ~40s/张 |
| 峰值显存 | ~2.2GB |
| LoRA 切换时间 | ~2s |

---

## 9. 部署架构

### 9.1 运行环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.10（3.11+ 与 torch 2.4~2.6 存在兼容问题） |
| Conda 环境 | 项目配置环境名 `costume310`（可自定义） |
| CUDA | 12.4（或 11.8/12.1，按 PyTorch 版本选择） |

`run_web.py` 内置环境检测逻辑：启动时检查 Python 版本和 CUDA 可用性，若不满足自动查找 `costume310` conda 环境的 Python 并重新启动。

### 9.2 单机部署

```
[用户浏览器] ---> localhost:8080 ---> [Gradio / FastAPI]
                                          │
                          ┌───────────────┼───────────────┐
                     [ResNet-50]    [SD 1.5 + LoRA]   [EasyNegative TI]
```

所有组件运行在单进程中，模型在启动时延迟加载（首次请求触发）。

### 9.2 目录与路径约定

| 路径 | 说明 |
|------|------|
| `sd/` | SD 1.5 diffusers 格式模型文件 |
| `models/classifier.pth` | ResNet-50 分类权重 |
| `models/lora/*.safetensors` | LoRA 权重文件 |
| `models/embeddings/EasyNegativeV2.safetensors` | TI 负向嵌入 |
| `outputs/samples/` | 生成图片默认保存位置 |
| `outputs/logs/` | 训练和推理日志 |
| `outputs/figures/` | 评估图表输出 |

---

## 10. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.3 | 2026-06-16 | AI 绘图新增 8 种采样器选择；代码清理（移除死导入、修正确路径）；文档修正 |
| 1.0.2 | 2026-06-16 | AI 绘图新增用户自定义正向/负向提示词输入框 |
| 1.0.1 | 2026-06-16 | run_web.py 增加环境检测与自动切换；修复 torchvision 循环导入问题 |
| 1.0.0 | 2026-06-16 | 初始版本：完成数据采集→分类→LoRA→Web 全管线 |
