# 源代码文件说明

本文档逐一说明项目中每个 `.py` 文件的作用及其关键代码逻辑。

---

## 目录

- [一、项目入口脚本](#一项目入口脚本)
- [二、数据采集模块 (src/data_collection/)](#二数据采集模块-srcdata_collection)
- [三、数据预处理模块 (src/data_preprocessing/)](#三数据预处理模块-srcdata_preprocessing)
- [四、图像分类模块 (src/classification/)](#四图像分类模块-srcclassification)
- [五、LoRA 训练模块 (src/lora_training/)](#五lora-训练模块-srclora_training)
- [六、Web 演示模块 (src/web/)](#六web-演示模块-srcweb)
- [七、工具脚本 (scripts/)](#七工具脚本-scripts)

---

## 一、项目入口脚本

### run.py — 一键全流程入口

**作用**：串联数据采集 → 预处理 → 训练 → Web 服务的完整流程，支持分步执行。

| 命令行参数 | 作用 |
|------------|------|
| `--data-only` | 仅执行数据采集与预处理 |
| `--train-only` | 仅执行分类模型训练 |
| `--web-only` | 仅启动 Web 服务 |
| `--target N` | 每类目标图片数量（默认 200） |

**关键代码**：

```python
def step_data_collection(target: int = 200):
    from src.data_collection.scraper import run_collection
    run_collection(base_dir=..., target_count=target)
```
按顺序调用各模块的主函数，每个 `step_*` 函数封装一个阶段的逻辑。`main()` 通过 `argparse` 解析参数，根据 `--data-only` / `--train-only` / `--web-only` 决定执行哪些步骤。

---

### run_data.py — 数据采集与预处理入口

**作用**：独立执行数据采集 + 清洗 + 划分 + 标注生成，不依赖其他模块。

**关键代码**：

```python
# 步骤1: 爬虫采集
from src.data_collection.scraper import run_collection
run_collection(base_dir=..., target_count=args.target)

# 步骤2: 数据清洗
from src.data_preprocessing.clean import run_cleaning_pipeline
run_cleaning_pipeline(raw_dir=..., cleaned_dir=..., blur_threshold=args.blur)

# 步骤3: 合并分类目录, 执行划分
from src.data_preprocessing.split import split_dataset
split_dataset(str(merged_dir), str(PROJECT_ROOT / "dataset" / "split"))

# 步骤4: 生成标注CSV
from src.data_collection.dataset_utils import generate_annotations_csv
generate_annotations_csv(raw_dir, csv_path)
```

流程说明：先爬虫下载图片到 `dataset/raw/<民族>/`，然后清洗输出到 `dataset/cleaned/<民族>_cls224/`（分类用 224×224）和 `<民族>_lora512/`（LoRA 用 512×512）。接着合并分类目录，按 70/15/15 划分，最后生成 `annotations.csv` 标注文件。

---

### run_train.py — 模型训练入口

**作用**：独立执行分类模型训练或 LoRA 训练，支持命令行参数调节。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--type` | classifier | classifier / lora / all |
| `--batch` | 32 | 分类训练批次大小 |
| `--epochs1` | 10 | 阶段1（冻结主干）轮数 |
| `--epochs2` | 20 | 阶段2（全网络微调）轮数 |
| `--lora-steps` | 1500 | LoRA 训练步数 |
| `--lora-ethnicity` | all | LoRA 训练的民族 |

**关键代码**：

```python
if args.type in ("classifier", "all"):
    from src.classification.train import train_model
    train_model(data_dir=..., model_save_path=...)
    # 训练完后自动评估
    from src.classification.evaluate import run_evaluation
    run_evaluation(...)

if args.type in ("lora", "all"):
    from src.lora_training.train_lora import train_lora, train_all
    if args.lora_ethnicity == "all":
        train_all(...)
    else:
        train_lora(args.lora_ethnicity, ...)
```

---

### run_web.py — Web 服务入口

**作用**：启动 Gradio + FastAPI Web 演示系统，**内置环境检测与自动切换**。

**关键代码——环境检测**（第 31-71 行）：

```python
def _is_valid_env():
    """检查 Python 3.10 + CUDA 可用"""
    py_ver = sys.version_info[:2]
    if py_ver != (3, 10):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    return True

def _relaunch_in_env():
    """查找 costume310 conda 环境的 Python 并重新启动自身"""
    conda_py = _find_conda_python()
    args = [str(conda_py), __file__] + sys.argv[1:]
    result = subprocess.run(args)
    sys.exit(result.returncode)

if not _is_valid_env():
    _relaunch_in_env()  # 自动切换到正确环境
```

这是解决 torchvision 循环导入问题的关键机制——如果用户在 base 环境（Python 3.13）运行，脚本会自动查找 `costume310` conda 环境的 Python 并用其重新执行自身。

**启动逻辑**（`main()`）：

```python
if args.api_only:
    uvicorn.run(app, host="0.0.0.0", port=args.port)  # 纯 API 模式
else:
    from src.web.app import launch_app
    launch_app(server_port=args.port, share=args.share)  # Gradio 模式
```

---

## 二、数据采集模块 (src/data_collection/)

### scraper.py — 网络图片爬虫

**作用**：从必应/百度图片搜索下载民族服饰图片，支持多线程并发下载和失败重试。

**关键代码——搜索关键词配置**（第 26-45 行）：

```python
ETHNIC_KEYWORDS = {
    "zang": ["藏族服饰", "藏族传统服装", "藏族节日盛装",
             "Tibetan traditional costume", "Tibetan dress", ...],
    "qiang": ["羌族服饰", "羌族传统服装", "羌族刺绣", ...],
    "yi":   ["彝族服饰", "彝族传统服装", "彝族银饰", ...],
}
```

每个民族配置了中英文混合的关键词列表，确保搜索结果多样性。

**关键代码——Bing 图片搜索**（`search_bing_images`，第 90-130 行）：

```python
base_url = f"https://www.bing.com/images/search?q={encoded_query}&first="
for offset in range(0, max_results * 2, 35):  # Bing 每页约35张
    resp = requests.get(base_url + str(offset), headers=HEADERS)
    matches = re.findall(r'"murl"\s*:\s*"([^"]+)"', resp.text)
    image_urls.extend(matches)
```

通过正则提取 Bing 页面中的 `"murl":"..."` 字段获取原图 URL。

**关键代码——并发下载**（`download_images`，第 175-251 行）：

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_download_image, url, fpath): (url, fpath)
               for url, fpath in tasks}
    for future in as_completed(futures):
        ok = future.result()
        if ok and _is_valid_image(fpath):
            success_count += 1
```

使用 `ThreadPoolExecutor` 实现 6 线程并发下载，`_download_image` 内建 3 次重试机制，`_is_valid_image` 通过 PIL 验证下载文件是否为有效图片。

**关键代码——主流程**（`run_collection`，第 254-283 行）：

```python
for ethnicity, keywords in ETHNIC_KEYWORDS.items():
    current_count = len(list(save_dir.glob("*")))
    needed = max(0, target_count - current_count)
    per_kw = max(10, needed // len(keywords) + 5)
    download_images(keywords, str(save_dir), max_per_keyword=per_kw)
```

对每个民族检查已有图片数量，只补充不足的部分，避免重复下载。

---

### dataset_utils.py — 数据集工具

**作用**：提供标签映射、数据集扫描统计、标注文件生成、图片验证等通用工具。

**关键代码——标签映射**（第 14-22 行）：

```python
LABEL_MAP = {
    "zang": 0, "qiang": 1, "yi": 2,
    0: "藏族", 1: "羌族", 2: "彝族",
}
```

双向映射：目录名（英文）↔ 数字 ID ↔ 中文名。

**关键代码——扫描数据集**（`scan_dataset`，第 39-66 行）：

```python
for ethnicity_dir in base_path.iterdir():
    if not ethnicity_dir.is_dir(): continue
    label = ethnicity_dir.name
    for img_file in ethnicity_dir.iterdir():
        if img_file.suffix.lower() in valid_extensions:
            records.append({"image_path": ..., "label": label, "filename": ...})
```

按照目录名自动推断标签（约定 `zang/`、`qiang/`、`yi/` 子目录 = 对应类别）。

**关键代码——生成标注 CSV**（`generate_annotations_csv`，第 69-107 行）：

```python
data.append({"image_path": ..., "label": ..., "source": "web_crawl",
             "quality_score": 1, "annotator": annotator,
             "annotation_date": today})
df.to_csv(output_path, index=False, encoding="utf-8-sig")
```

输出 UTF-8-BOM 编码的 CSV（确保 Excel 直接打开不乱码）。

---

## 三、数据预处理模块 (src/data_preprocessing/)

### clean.py — 数据清洗

**作用**：对原始图片执行去重、模糊检测、尺寸统一三阶段清洗流程。

**关键代码——感知哈希去重**（`remove_duplicates`，第 22-66 行）：

```python
h = compute_phash(str(f))  # imagehash.phash
# 两两比较汉明距离
dist = sum(c1 != c2 for c1, c2 in zip(hash_list[i][1], hash_list[j][1]))
if dist < threshold:  # 默认 5
    removed.add(hash_list[j][0])  # 移除后者
```

使用 `imagehash.phash` 计算感知哈希，通过汉明距离比较相似度。阈值 5 表示：两张图片视觉差异小于 5 个哈希位视为重复。

**关键代码——拉普拉斯模糊检测**（`detect_blur_laplacian`，第 69-84 行）：

```python
img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
return laplacian_var < threshold, laplacian_var
```

对灰度图计算拉普拉斯算子的方差。清晰图片边缘丰富，方差大；模糊图片方差小。默认阈值 100。

**关键代码——尺寸调整**（`resize_images`，第 113-158 行）：

```python
if keep_aspect:
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(img, (paste_x, paste_y))  # 居中 + 黑边填充
else:
    new_img = img.resize(target_size, Image.Resampling.LANCZOS)  # 直接缩放
```

分类图片直接缩放到 224×224，LoRA 图片保持宽高比后用黑边填充到 512×512。

**关键代码——完整清洗流程**（`run_cleaning_pipeline`，第 161-212 行）：

```python
# 对每个民族:
# 1. 复制到临时目录
# 2. remove_duplicates() → 3. remove_blurry()
# 4. resize_images(cls_out, 224×224) → 5. resize_images(lora_out, 512×512, keep_aspect=True)
# 6. 清理临时目录
```

输出两套数据：`<民族>_cls224/` 供分类训练，`<民族>_lora512/` 供 LoRA 训练。

---

### split.py — 数据集划分

**作用**：按 70/15/15 比例划分训练集、验证集、测试集，使用分层抽样保证各类别比例一致。

**关键代码**（`split_dataset`，第 13-97 行）：

```python
random.seed(seed)
random.shuffle(images)
n_train = int(n_total * train_ratio)    # 70%
n_val = int(n_total * val_ratio)         # 15%
# 剩余归测试集

train_images = images[:n_train]
val_images = images[n_train:n_train + n_val]
test_images = images[n_train + n_val:]
```

对每个类别分别 shuffle 后按比例切分，保证每个子集中各类别比例与原始分布一致（分层抽样）。固定 `seed=42` 确保划分可复现。

---

### augment.py — 数据增强

**作用**：为分类训练提供离线的数据增强变换（水平翻转、旋转、颜色抖动、随机裁剪）。

**关键代码——四种增强变换**：

```python
random_horizontal_flip(img, p=0.5)       # 50% 概率水平翻转
random_rotation(img, max_degrees=15)      # ±15° 随机旋转
random_color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2)
random_resized_crop(img, (224,224), scale=(0.8, 1.0))
```

`random_color_jitter` 使用 PIL 的 `ImageEnhance` 分别对亮度、对比度、饱和度施加 ±20% 的随机扰动。`random_resized_crop` 模拟 torchvision 的 `RandomResizedCrop`：随机选区域（面积 80%~100%），裁剪后缩放。

**注意**：训练时实际使用的是 `train.py` 中 `build_transforms()` 的在线增强（`torchvision.transforms`），本模块的离线增强主要用于在训练前扩充数据集。

---

## 四、图像分类模块 (src/classification/)

### train.py — 分类模型训练

**作用**：基于 ResNet-50 的两阶段迁移学习训练，含早停和学习率调度。

**关键代码——模型构建**（`create_model`，第 60-70 行）：

```python
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, num_classes),
)
```

加载 ImageNet 预训练权重，替换最后的全连接层为 Dropout(0.3) + Linear(2048→3)。

**关键代码——两阶段训练**（`train_model`，第 140-288 行）：

```python
# 阶段1: 冻结主干
freeze_backbone(model)  # 除 fc 层外全部 requires_grad=False
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
# 训练 10 epochs

# 阶段2: 解冻全网络
unfreeze_all(model)
optimizer = Adam(model.parameters(), lr=1e-4)  # 更低学习率
# 训练 20 epochs
```

阶段1 只训练新替换的分类头（快速收敛），阶段2 以低学习率微调整个网络（精细调优）。

**关键代码——学习率调度与早停**（第 186-214 行）：

```python
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
scheduler.step(val_loss)

if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), model_save_path)
else:
    patience_counter += 1
    if patience_counter >= EARLY_STOP_PATIENCE:  # 5 epochs
        break
```

当验证 loss 连续 3 epoch 不降时学习率折半；验证准确率连续 5 epoch 不提升时停止训练。

---

### evaluate.py — 模型评估

**作用**：生成混淆矩阵热力图、ROC 曲线、训练历史曲线、文本评估报告。

**关键代码——混淆矩阵**（`plot_confusion_matrix`，第 25-42 行）：

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
```

使用 seaborn 绘制带数值标注的热力图。

**关键代码——ROC 曲线**（`plot_roc_curves`，第 45-78 行）：

```python
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(y_scores)[:, i])
    roc_auc = auc(fpr, tpr)
```

使用 One-vs-Rest 策略绘制多分类 ROC 曲线——将每个类别依次当作正类，其余为负类。

**关键代码——命令行入口修复**（第 220 行）：

```python
run_evaluation(str(preds_abs), str(project_root / args.model), str(output_abs))
```

将 `args.model`（可能是相对路径）拼接为绝对路径，避免因当前工作目录与项目根目录不一致导致的路径错误。

---

### inference.py — 分类推理接口

**作用**：提供 `CostumeClassifier` 类，封装模型加载、图片预处理和推理逻辑，支持全局单例模式。

**关键代码——类设计**（`CostumeClassifier`，第 22-109 行）：

```python
class CostumeClassifier:
    def __init__(self, model_path):
        self.model = self._build_model()   # 构建与训练时一致的结构
        self._load_weights()               # 加载权重
        self.model.eval()

    def predict(self, image_path, top_k=3):
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        # 返回 top-k 结果
```

**关键代码——预处理管道**（第 38-43 行）：

```python
self.transform = transforms.Compose([
    transforms.Resize(256),              # 短边缩放到 256
    transforms.CenterCrop(224),          # 中心裁剪 224×224
    transforms.ToTensor(),               # 转为 Tensor + 归一化到 [0,1]
    transforms.Normalize(MEAN, STD),     # ImageNet 均值/标准差归一化
])
```

必须与训练时的验证集预处理完全一致，否则推理结果不准确。

**关键代码——全局单例**（第 116-121 行）：

```python
_classifier_instance: CostumeClassifier | None = None

def get_classifier(model_path):
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = CostumeClassifier(model_path)
    return _classifier_instance
```

延迟加载模式：首次调用时才加载模型（~97MB），之后复用同一实例，避免 Web 服务的每个请求都重新加载。

---

## 五、LoRA 训练模块 (src/lora_training/)

### config.py — 训练配置

**作用**：定义 LoRA 训练和推理的所有超参数、民族触发词、提示词模板。

**关键代码——民族配置**（`ETHNIC_CONFIGS`，第 21-64 行）：

```python
"zang": {
    "name": "藏族",
    "trigger_word": "zang_costume",  # LoRA 触发词
    "instance_prompt": "zang_costume, traditional Tibetan robe, ...",
    "description_keywords": ["traditional Tibetan robe", "colorful stripes", ...],
},
```

每个民族定义了触发词（LoRA 激活词）、实例提示词（训练用）、描述关键词列表（用于自动生成训练数据的文本描述）。

**关键代码——训练超参数**（`TRAINING_CONFIG`，第 70-106 行）：

```python
"lora_rank": 16,       # 低秩矩阵维度
"lora_alpha": 32,       # 缩放因子 (scale = alpha/r = 2.0)
"lora_dropout": 0.05,
"lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0",
                        "ff.net.0.proj", "ff.net.2"],
"learning_rate": 1e-4,
"lr_scheduler": "cosine",
"max_train_steps": 1500,
```

`lora_target_modules` 指定了 LoRA 生效的 UNet 层——同时覆盖注意力投影层（to_q/k/v/out）和前馈网络层（ff.net），确保风格信息能充分注入。

---

### prepare_dataset.py — LoRA 数据准备

**作用**：将清洗后的图片处理为 LoRA 训练所需的 512×512 图片 + 同名 .txt 描述文件对。

**关键代码——图片处理 + 文本生成**（`prepare_instance_images`，第 16-77 行）：

```python
# 居中裁剪为正方形
w, h = img.size
s = min(w, h)
img = img.crop((left, top, left + s, top + s))
img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

# 生成随机描述
keywords = random.sample(config["description_keywords"], min(5, len(...)))
caption = f"{config['trigger_word']}, {', '.join(keywords)}"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(caption)
```

每张图片从关键词库中随机抽取 5 个关键词，与触发词组合生成唯一的文本描述。这样每张图的描述略有不同，增加了训练数据的多样性。

**关键代码——源目录搜索**（第 102-113 行）：

```python
search_dirs = [
    source_path / ethnicity,
    source_path / f"{ethnicity}_lora512",
    source_path / f"{ethnicity}_cls224",
]
```

自动在 `raw/`、`cleaned/<民族>_lora512/`、`cleaned/<民族>_cls224/` 中搜索可用数据源。

---

### train_lora.py — LoRA 训练

**作用**：基于 diffusers + peft 库实现 LoRA 微调，包含完整训练循环、检查点保存和学习率余弦退火。

**关键代码——数据集类**（`LoraDataset`，第 22-62 行）：

```python
class LoraDataset(Dataset):
    def __getitem__(self, idx):
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.tensor(img_np, ...).reshape(512,512,3) / 127.5 - 1.0
        # 读取同名 .txt 作为 prompt
        txt_path = img_path.with_suffix(".txt")
        prompt = open(txt_path).read().strip() if txt_path.exists() else "..."
        return {"pixel_values": img_tensor, "prompt": prompt}
```

图片归一化到 [-1, 1]（SD 标准输入范围），prompt 从 prepare_dataset 阶段生成的 .txt 文件读取。

**关键代码——LoRA 配置与注入**（第 152-168 行）：

```python
lora_config = LoraConfig(
    r=train_cfg["lora_rank"],           # 16
    lora_alpha=train_cfg["lora_alpha"], # 32
    lora_dropout=train_cfg["lora_dropout"],
    target_modules=train_cfg["lora_target_modules"],
)
unet = get_peft_model(unet, lora_config)  # 注入 LoRA 层
vae.requires_grad_(False)       # VAE 冻结
text_encoder.requires_grad_(False)  # CLIP 冻结
```

只有 UNet 的 LoRA 插入层参与训练，VAE 和 CLIP 文本编码器完全冻结。

**关键代码——训练循环核心**（第 233-268 行）：

```python
# 编码图片到 latent 空间
with torch.no_grad():
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor  # 0.18215

# 添加噪声
noise = torch.randn_like(latents)
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

# UNet 预测噪声
noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
loss = F.mse_loss(noise_pred, noise)  # 预测噪声 vs 真实噪声的 MSE

# 梯度累积
loss.backward()
if (global_step + 1) % gradient_accumulation_steps == 0:
    clip_grad_norm_(unet.parameters(), 1.0)
    optimizer.step()
    scheduler.step()   # 余弦退火
```

这是标准扩散模型训练流程：图片→VAE 编码→加噪→UNet 预测噪声→MSE Loss。每 4 步更新一次权重（梯度累积等效 batch_size=4）。

**关键代码——学习率余弦退火**（第 194-198 行）：

```python
lr_lambda = lambda step: max(
    0.0, 0.5 * (1.0 + math.cos(math.pi * step / max_train_steps)))
scheduler = LambdaLR(optimizer, lr_lambda)
```

学习率从初始值 `1e-4` 开始，沿余弦曲线平滑衰减至接近 0。

---

## 六、Web 演示模块 (src/web/)

### api.py — FastAPI 后端

**作用**：提供 RESTful API 接口，包括健康检查、图像分类、文生图三个核心端点。

**关键代码——请求/响应模型**（第 54-82 行）：

```python
class GenerateRequest(BaseModel):
    ethnicity: str = Field(default="zang")
    prompt: str = Field(default="")
    steps: int = Field(default=25, ge=10, le=50)       # 步数范围校验
    lora_scale: float = Field(default=0.8, ge=0.1, le=1.5)
    cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    custom_positive: str = Field(default="")
    custom_negative: str = Field(default="")
    sampler: str = Field(default="DPM++ 2M Karras")
```

Pydantic 的 `Field(ge=..., le=...)` 提供了自动参数范围校验。

**关键代码——分类接口**（`/classify`，第 137-161 行）：

```python
if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
    raise HTTPException(400, "请上传 JPEG/PNG/WEBP/BMP 格式的图片")
img = Image.open(io.BytesIO(contents)).convert("RGB")
classifier = get_classifier()
results = classifier.predict_from_pil(img, top_k=3)
```

先校验 MIME 类型，再通过 PIL 解码，最后调用 `CostumeClassifier.predict_from_pil()` 推理。

**关键代码——生成接口图片编码**（`/generate`，第 186-191 行）：

```python
import base64
buf = io.BytesIO()
img.save(buf, format="PNG")
image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
```

生成的 PIL Image 编码为 PNG 再转 Base64，便于在 JSON 响应中传输。

---

### app.py — Gradio 前端

**作用**：构建三 Tab 的 Web 交互界面（服饰识别 + AI 绘图 + 系统信息）。

**关键代码——界面结构**（`create_app`，第 88-265 行）：

```python
with gr.Blocks(title="民族服饰识别与AI绘图系统") as demo:
    with gr.Tabs():
        with gr.Tab("服饰识别"):
            img_input = gr.Image(type="pil", sources=["upload", "clipboard"])
            classify_btn.click(classify_fn, inputs=[img_input], outputs=[result_output])

        with gr.Tab("AI 绘图"):
            ethnicity_select = gr.Dropdown(choices=["zang","qiang","yi"], value="zang")
            sampler_select = gr.Dropdown(choices=[...8种采样器...])
            positive_input = gr.Textbox(lines=3)   # 自定义正向提示词
            negative_input = gr.Textbox(lines=3)   # 自定义负向提示词
            generate_btn.click(generate_fn, inputs=[...], outputs=[img_output, ...])

        with gr.Tab("系统信息"): ...
```

**关键代码——Gradio 启动参数**（`launch_app`，第 268-283 行）：

```python
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
demo.launch(
    server_name="127.0.0.1",   # Gradio 6.x 需显式绑 127.0.0.1
    server_port=server_port,
    ssr_mode=False,             # 禁用 SSR 避免 Windows 上的 502 错误
)
```

`ssr_mode=False` 是 Gradio 6.18.0 在 Windows 上的关键修复——SSR 模式下启动事件会返回 502。`no_proxy` 防止代理干扰 requests 库的本地请求。

---

### inference.py — SD+LoRA 推理引擎

**作用**：使用 diffusers 直接加载本地 SD 1.5 模型和 LoRA 权重进行文生图推理，**不依赖任何外部 WebUI**。

**关键代码——Pipeline 加载**（`_load_pipeline`，第 41-77 行）：

```python
pipe = StableDiffusionPipeline.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,          # GPU 用 FP16 省显存
    safety_checker=None,                # 跳过安全检查器，省 ~600MB
    local_files_only=True,              # 仅本地加载，不联网
)
pipe.enable_attention_slicing()         # 注意力分片
pipe.enable_vae_slicing()               # VAE 解码分片
pipe.load_textual_inversion(str(EMBEDDING_PATH), token="easynegative")
```

三行优化将显存占用从 ~4GB 降至 ~2.2GB。

**关键代码——采样器切换**（`_apply_scheduler`，第 80-121 行）：

```python
scheduler_cls = {
    "DPM++ 2M Karras": DPMSolverMultistepScheduler,
    "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    # ... 共 8 种
}.get(sampler_name)

kwargs = {}
if scheduler_cls is DPMSolverMultistepScheduler:
    kwargs["use_karras_sigmas"] = True
    kwargs["algorithm_type"] = "dpmsolver++"

pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
```

从已有 scheduler 的 config 创建新 scheduler 实例，确保配置兼容。每次切换只重建 scheduler（轻量操作），不重新加载整个 pipeline。

**关键代码——LoRA 加载与动态缩放**（`_apply_lora`，第 124-164 行）：

```python
# 如果已有 LoRA，先 merge_and_unload 还原
if isinstance(pipe.unet, PeftModel):
    pipe.unet = pipe.unet.merge_and_unload()

# 重新注入 LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=[...])
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.load_state_dict(state_dict, strict=False)

# 动态调整 LoRA 强度
for module in pipe.unet.modules():
    if hasattr(module, "scaling") and "default" in module.scaling:
        module.scaling["default"] = lora_scale
```

关键设计：切换民族时先 `merge_and_unload()` 还原原始 UNet，再用新 LoRA 权重重新注入。避免了每次推理都重新加载整个 pipeline（~10s → ~2s）。`module.scaling["default"]` 动态调整允许用户在界面滑块上实时改变 LoRA 强度而无需重新加载权重。

**关键代码——推理执行**（`generate_local`，第 167-229 行）：

```python
positive, negative = build_prompt(ethnicity, user_prompt, lora_scale,
                                  custom_positive, custom_negative)
pipe = _load_pipeline()
_apply_lora(pipe, ethnicity, lora_scale)
_apply_scheduler(pipe, sampler)

generator = torch.Generator(device=pipe.device).manual_seed(seed)
with torch.inference_mode():
    result = pipe(prompt=positive, negative_prompt=negative,
                  num_inference_steps=steps, guidance_scale=cfg_scale,
                  generator=generator, width=512, height=512)
```

先构建提示词 → 加载/切换模型 → 切换采样器 → 设置随机种子 → 执行推理。

---

### prompts.py — 提示词模板

**作用**：为三个民族定义预设的正/负向提示词模板，支持用户自定义覆盖。

**关键代码——提示词构建**（`build_prompt`，第 21-88 行）：

```python
# 正向提示词：优先使用自定义，否则用预设模板
if custom_positive.strip():
    positive = custom_positive.strip()
else:
    positive = template["base"]
    if user_description:
        positive = f"{positive}, {user_description}"

# 负向提示词：优先使用自定义，否则用默认
negative = custom_negative.strip() if custom_negative.strip() else NEGATIVE_PROMPT
```

优先级：自定义 → 预设模板。负向提示词默认注入 `easynegative`（EasyNegativeV2 嵌入的触发词）。

**关键代码——预设模板示例**（第 38-71 行）：

```python
"zang": {
    "trigger": "zang_costume",
    "base": "zang_costume, traditional Tibetan robe, colorful stripes, "
            "fur trim, silver jewelry, plateau landscape background, "
            "masterpiece, best quality, 8k, detailed fabric texture",
},
```

每个民族的模板包含触发词 + 核心视觉特征 + 质量修饰词（masterpiece, 8k 等）。

---

## 七、工具脚本 (scripts/)

### test_model_load.py — SD 模型完整性验证

**作用**：逐一加载 SD 1.5 的四个组件（tokenizer、text_encoder、UNet、scheduler），验证模型文件完整且可读取。

```python
tok = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet")
sched = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
```

如果任一组件加载失败，说明模型文件不完整或损坏。全部成功则表示 `sd/` 目录可用。

---

### test_embedding.py — EasyNegative 嵌入测试

**作用**：测试 EasyNegativeV2 文本反转嵌入能否随 pipeline 正确加载。

```python
from web.inference import _load_pipeline
pipe = _load_pipeline()  # 内部调用 pipe.load_textual_inversion(...)
```

验证 `models/embeddings/EasyNegativeV2.safetensors` 文件存在且格式正确。

---

### test_inference.py — SD+LoRA 推理测试

**作用**：端到端测试本地 SD + LoRA 推理是否正常工作，生成一张藏族风格样图。

```python
pipe = _load_pipeline()
_apply_lora(pipe, "zang", 0.8)
with torch.inference_mode():
    result = pipe(prompt=positive, negative_prompt=negative,
                  num_inference_steps=20, guidance_scale=7.0)
img.save("outputs/samples/test_zang_lora.png")
```

如果成功生成保存图片，则整个推理链路（模型加载→LoRA 注入→采样→保存）验证通过。

---

### download_sd15.py — SD 模型下载脚本

**作用**：通过 ModelScope SDK 下载 SD 1.5 模型（替代 HuggingFace）。

```python
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/stable-diffusion-v1-5', cache_dir=...)
```

适用于网络能访问 ModelScope 的环境。注意：此脚本硬编码了本地路径，使用前需修改 `cache_dir`。

---

### generate_docx.py — 项目文档生成器

**作用**：根据当前项目状态自动生成 `.docx` 格式的项目文档。使用 `python-docx` 库构建包含封面、目录、10 个章节、14 个表格的格式化文档。

---

### \_\_init\_\_.py — 包初始化文件

项目中共有 6 个 `__init__.py`：

| 位置 | 作用 |
|------|------|
| `src/__init__.py` | 标记 `src/` 为 Python 包，含一行项目名称注释 |
| `src/data_collection/__init__.py` | 标记为包（空文件） |
| `src/data_preprocessing/__init__.py` | 标记为包（空文件） |
| `src/classification/__init__.py` | 标记为包（空文件） |
| `src/lora_training/__init__.py` | 标记为包（空文件） |
| `src/web/__init__.py` | 标记为包（空文件） |

这些文件使 Python 能将上述目录识别为包，从而支持 `from src.xxx import yyy` 和 `python -m src.xxx.yyy` 两种调用方式。模块间的相对导入（如 `from .config import ...`）也依赖它们。
