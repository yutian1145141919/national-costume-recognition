"""生成项目文档 .docx 文件"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "民族服饰识别与AI绘图项目文档.docx"


def set_cell_shading(cell, color):
    """设置单元格底色"""
    shading_elm = cell._element.get_or_add_tcPr()
    shading = shading_elm.makeelement(qn("w:shd"), {
        qn("w:fill"): color,
        qn("w:val"): "clear",
    })
    shading_elm.append(shading)


def add_styled_table(doc, headers, rows, col_widths=None):
    """添加带样式的表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
        set_cell_shading(cell, "D9E2F3")

    # 数据行
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    doc.add_paragraph("")
    return table


def build_docx():
    doc = Document()

    # 页面设置
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

    # ========== 封面 ==========
    for _ in range(6):
        doc.add_paragraph("")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("民族服饰识别与AI绘图综合实践项目")
    run.bold = True
    run.font.size = Pt(26)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Project Documentation")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x59, 0x56, 0x59)

    doc.add_paragraph("")

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run("适用专业：数据科学与大数据技术\n项目性质：综合实践 / 课程设计\n建议学时：6~8 周\n版本：1.0.3\n日期：2026-06-16")
    run.font.size = Pt(11)

    doc.add_page_break()

    # ========== 目录页 ==========
    doc.add_heading("目录", level=1)
    toc_items = [
        "一、项目概述",
        "  1.1 项目背景",
        "  1.2 项目目标",
        "  1.3 核心功能",
        "二、技术架构",
        "  2.1 系统架构",
        "  2.2 技术栈选型",
        "  2.3 项目结构",
        "三、环境配置",
        "  3.1 最低配置要求",
        "  3.2 环境搭建步骤",
        "四、数据采集与预处理",
        "  4.1 数据采集",
        "  4.2 数据清洗",
        "  4.3 数据集划分",
        "  4.4 数据增强",
        "五、图像分类模型",
        "  5.1 模型选型与训练策略",
        "  5.2 训练参数配置",
        "  5.3 模型评估与实际结果",
        "六、LoRA 风格模型",
        "  6.1 LoRA 原理简介",
        "  6.2 训练数据准备",
        "  6.3 训练配置与参数",
        "  6.4 训练结果",
        "七、Web 演示系统",
        "  7.1 系统功能",
        "  7.2 API 接口规格",
        "  7.3 推理优化策略",
        "八、部署流程",
        "九、项目成果与交付",
        "十、常见问题与故障排除",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # ============================================================
    # 一、项目概述
    # ============================================================
    doc.add_heading("一、项目概述", level=1)

    doc.add_heading("1.1 项目背景", level=2)
    doc.add_paragraph(
        "本项目以少数民族非物质文化遗产保护为背景，融合计算机视觉、深度学习与生成式 AI 等前沿技术，"
        "围绕藏族、羌族、彝族三个民族的传统服饰图像，构建一套完整的「数据采集 → 模型训练 → 智能识别 → AI 绘图」闭环系统。"
    )
    doc.add_paragraph(
        "藏族、羌族、彝族均分布于中国西南地区，各自拥有独特而丰富的服饰文化，是中华民族多元文化的重要组成部分。"
        "借助 AI 技术对这些传统服饰进行自动识别与风格化生成，有助于文化遗产的数字化保存与创新传播，"
        "同时也为数据科学专业学生提供了一个综合性的实践平台。"
    )

    doc.add_heading("1.2 项目目标", level=2)
    doc.add_paragraph("通过本项目，学生将系统性地掌握以下能力：")
    targets = [
        "图像数据采集规范与标注流程，理解数据工程质量控制的重要性",
        "经典图像识别模型（ResNet-50）的训练、调参与评估方法，掌握迁移学习与计算机视觉基础",
        "LoRA（Low-Rank Adaptation）参数化微调技术在图像生成模型（AIGC）中的应用",
        "将训练好的模型封装为 Web API 服务，构建可交互的演示前端",
        "撰写完整的技术文档与项目报告",
    ]
    for t in targets:
        doc.add_paragraph(t, style="List Bullet")
    doc.add_paragraph(
        "学科交叉覆盖：本项目涉及 Python 编程、数据预处理、深度学习、模型评估、Web 服务部署等多个知识模块，"
        "是数据科学专业综合能力训练的典型载体。"
    )

    doc.add_heading("1.3 核心功能", level=2)
    add_styled_table(doc,
        ["功能模块", "说明", "技术实现"],
        [
            ["数据采集", "自动从公开图片源搜索并下载民族服饰图片", "requests + BeautifulSoup"],
            ["数据预处理", "感知哈希去重、模糊检测、尺寸统一、数据增强", "OpenCV + imagehash"],
            ["图像识别", "基于 ResNet-50 的三分类模型，测试准确率 88%", "PyTorch + torchvision"],
            ["LoRA 风格模型", "基于 Stable Diffusion 1.5 的民族服饰风格微调", "diffusers + peft"],
            ["Web 演示", "三 Tab 界面：上传识别 + 文生图 + 系统信息", "FastAPI + Gradio"],
        ],
        col_widths=[3, 6, 5]
    )

    # ============================================================
    # 二、技术架构
    # ============================================================
    doc.add_heading("二、技术架构", level=1)

    doc.add_heading("2.1 系统架构", level=2)
    doc.add_paragraph("本系统采用分层架构设计，各层职责明确、耦合度低：")

    arch_items = [
        "数据层：原始图像 → 标注 → 清洗 → 划分（训练/验证/测试），同时为 LoRA 训练准备 512×512 图像与文本描述对",
        "模型层：ResNet-50（分类推理）+ Stable Diffusion 1.5 + LoRA（图像生成）+ EasyNegative 文本反转嵌入",
        "服务层：FastAPI 后端，提供 /classify（图像识别）、/generate（文生图）、/health（健康检查）接口",
        "展示层：Gradio 前端，包含服饰识别、AI 绘图、系统信息三个 Tab 页",
    ]
    for item in arch_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("2.2 技术栈选型", level=2)
    add_styled_table(doc,
        ["层级", "推荐方案", "说明"],
        [
            ["深度学习框架", "PyTorch 2.4.0 + torchvision 0.19.0", "GPU 加速训练与推理"],
            ["分类模型", "ResNet-50（ImageNet 预训练）", "迁移学习，适合小数据集微调"],
            ["生成模型", "Stable Diffusion 1.5 + LoRA（diffusers）", "本地推理，不依赖 SD WebUI"],
            ["LoRA 训练", "diffusers + peft", "纯 Python 方案，比 kohya_ss 更轻量"],
            ["后端服务", "FastAPI + Uvicorn", "高性能异步 REST API"],
            ["前端展示", "Gradio 6.x", "快速构建机器学习演示界面"],
            ["数据处理", "OpenCV + Pillow + scikit-learn", "图像处理与评估指标"],
        ],
        col_widths=[3.5, 5.5, 5]
    )

    doc.add_heading("2.3 项目结构", level=2)
    doc.add_paragraph("项目采用模块化组织，按功能域划分源码目录：")
    structure = """民族lora/
├── CLAUDE.md                   # 项目协作规范
├── README.md                   # 项目说明
├── requirements.txt            # Python 依赖清单
├── run.py                      # 一键全流程入口
├── run_data.py                 # 数据采集+预处理入口
├── run_train.py                # 模型训练入口
├── run_web.py                  # Web 服务入口（含环境检测）
├── docs/                       # 项目文档
│   ├── deployment.md           # 部署流程操作文档
│   ├── HELP.md                 # 详细使用帮助
│   └── specification.md        # 技术规格文档
├── libs/                       # 运行时依赖库
│   └── libomp140.x86_64.dll   # LLVM OpenMP 运行时
├── sd/                         # Stable Diffusion 1.5 模型文件
├── dataset/                    # 数据集目录
│   ├── raw/                    # 原始图片（zang/qiang/yi）
│   ├── cleaned/                # 清洗后图片
│   ├── split/                  # 数据集划分（train/val/test）
│   ├── lora_data/              # LoRA 训练数据
│   └── annotations.csv         # 标注记录
├── models/                     # 模型存储
│   ├── classifier.pth          # 分类模型权重 (~97MB)
│   ├── lora/                   # LoRA 权重文件（.safetensors）
│   └── embeddings/             # 文本反转嵌入
│       └── EasyNegativeV2.safetensors
├── src/                        # 源代码
│   ├── data_collection/        # 数据采集模块
│   ├── data_preprocessing/     # 数据预处理模块
│   ├── classification/         # 图像分类模块
│   ├── lora_training/          # LoRA 训练模块
│   └── web/                    # Web 演示模块
├── outputs/                    # 输出结果
│   ├── figures/                # 评估图表
│   ├── logs/                   # 训练日志
│   └── samples/                # 生成样图
└── scripts/                    # 工具脚本"""
    p = doc.add_paragraph()
    run = p.add_run(structure)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)

    # ============================================================
    # 三、环境配置
    # ============================================================
    doc.add_heading("三、环境配置", level=1)

    doc.add_heading("3.1 最低配置要求", level=2)
    add_styled_table(doc,
        ["组件", "最低要求", "推荐配置", "备注"],
        [
            ["操作系统", "Windows 10+ / Linux / macOS", "Windows 10+", "本项目在 Windows 10 上开发测试"],
            ["Python", "3.10", "3.10", "3.11 及以上与 torch 2.4 有兼容问题"],
            ["GPU 显存", "3GB（仅推理）/ 6GB（含训练）", "8GB+", "CPU 也可运行但速度较慢"],
            ["CUDA", "12.4（或 11.8/12.1）", "12.4", "需与 PyTorch 版本匹配"],
            ["内存", "8GB", "16GB+", ""],
            ["磁盘", "15GB", "20GB+ (SSD)", "SD 模型 ~4GB + 依赖 ~3GB + 数据 ~2GB"],
        ],
        col_widths=[2.5, 4, 3, 4.5]
    )

    doc.add_heading("3.2 环境搭建步骤", level=2)
    steps = [
        ("步骤 1：创建虚拟环境",
         "使用 Conda 创建 Python 3.10 环境：\n"
         "  conda create -n costume310 python=3.10 -y\n"
         "  conda activate costume310"),
        ("步骤 2：安装 PyTorch",
         "在线安装（推荐）：\n"
         "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n\n"
         "或手动下载 whl 文件安装（网络不稳定时）：\n"
         "  pip install torch-2.4.0+cu124-cp310-cp310-win_amd64.whl\n"
         "  pip install torchvision-0.19.0+cu124-cp310-cp310-win_amd64.whl\n\n"
         "验证：python -c \"import torch; print(torch.cuda.is_available())\""),
        ("步骤 3：解决 Windows DLL 依赖",
         "如遇 fbgemm.dll 加载失败：\n"
         "  1) 安装 VC++ Redist (vc_redist.x64.exe)\n"
         "  2) 将 libs/libomp140.x86_64.dll 复制到 conda 环境的 Lib/site-packages/torch/lib/"),
        ("步骤 4：安装项目依赖",
         "  pip install -r requirements.txt\n\n"
         "注意：安装 Gradio 后可能需降级 huggingface-hub：\n"
         "  pip install \"huggingface-hub<1.0,>=0.23.2\"\n"
         "  pip install \"numpy<2\""),
        ("步骤 5：下载 Stable Diffusion 1.5 模型",
         "从 ModelScope（魔搭社区）下载：\n"
         "  git clone https://www.modelscope.cn/AI-ModelScope/stable-diffusion-v1-5.git sd\n\n"
         "或访问 ModelScope 文件页手动下载所需文件。\n"
         "验证：python scripts/test_model_load.py"),
        ("步骤 6：放置 EasyNegative 嵌入（可选）",
         "将 EasyNegativeV2.safetensors 放入 models/embeddings/ 目录，\n"
         "推理时自动加载作为负向提示词前缀。"),
    ]
    for title, content in steps:
        doc.add_heading(title, level=3)
        p = doc.add_paragraph()
        run = p.add_run(content)
        run.font.size = Pt(10)

    doc.add_paragraph(
        "关键依赖版本对照：torch 2.4.0+cu124、torchvision 0.19.0+cu124、diffusers 0.31.0、"
        "peft 0.19.1、transformers 4.44.2、gradio 6.x、numpy <2.0、huggingface-hub 0.36.x。"
        "详细环境配置说明见 docs/deployment.md。"
    )

    # ============================================================
    # 四、数据采集与预处理
    # ============================================================
    doc.add_heading("四、数据采集与预处理", level=1)

    doc.add_heading("4.1 数据采集", level=2)
    doc.add_heading("4.1.1 数据来源", level=3)
    sources = [
        '网络爬虫：必应图片搜索（Bing Image Search），使用「藏族服饰」「羌族服饰」「彝族服饰」等关键词',
        "开放数据集：Kaggle 等平台的民族服饰相关数据集（补充来源）",
        "手动采集：博物馆官网、文化机构的公开展示图片（可选加分项）",
    ]
    for s in sources:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("4.1.2 图像质量规范", level=3)
    add_styled_table(doc,
        ["指标", "要求", "说明"],
        [
            ["最小分辨率", "短边 ≥ 256px", "低于此分辨率影响特征提取效果"],
            ["每类数量", "≥ 200 张（建议 500+）", "数量越多模型泛化能力越强"],
            ["格式", "JPG / PNG / BMP / WebP", "采集后统一处理"],
            ["内容", "以服饰为主体，背景尽量干净", "避免背景过于复杂的图片"],
            ["多样性", "覆盖不同性别、年龄、场景", "防止模型过拟合特定样式"],
            ["版权", "仅用于教学研究", "注意合规使用"],
        ],
        col_widths=[3, 4, 7]
    )

    doc.add_heading("4.1.3 数据目录结构", level=3)
    p = doc.add_paragraph()
    run = p.add_run(
        "dataset/raw/\n"
        "├── zang/    # 藏族服饰图片\n"
        "├── qiang/   # 羌族服饰图片\n"
        "└── yi/      # 彝族服饰图片"
    )
    run.font.name = "Consolas"
    run.font.size = Pt(10)

    doc.add_heading("4.2 数据清洗", level=2)
    doc.add_paragraph("数据清洗自动执行以下步骤：")
    clean_steps = [
        "感知哈希（pHash）去重：计算图片指纹，汉明距离 ≤ 5 视为重复并移除",
        "拉普拉斯模糊检测：计算拉普拉斯方差，阈值 < 100 视为模糊图片并移除",
        "尺寸统一：分类数据 Resize + CenterCrop 至 224×224；LoRA 数据 center-crop 至 512×512",
        "自动移除水印密集、损坏不可读的图片",
    ]
    for s in clean_steps:
        doc.add_paragraph(s, style="List Bullet")

    doc.add_heading("4.3 数据集划分", level=2)
    add_styled_table(doc,
        ["子集", "比例", "用途"],
        [
            ["train（训练集）", "70%", "模型参数训练"],
            ["val（验证集）", "15%", "超参数调优、早停判断"],
            ["test（测试集）", "15%", "最终评估（不参与训练）"],
        ],
        col_widths=[4, 2, 8]
    )
    doc.add_paragraph("划分方式：分层抽样（stratified split），各类别比例在各子集中保持一致。")

    doc.add_heading("4.4 数据增强（分类训练时）", level=2)
    add_styled_table(doc,
        ["变换", "参数", "说明"],
        [
            ["RandomHorizontalFlip", "p=0.5", "随机水平翻转"],
            ["RandomRotation", "±15°", "随机旋转"],
            ["ColorJitter", "brightness/contrast/saturation ±20%", "颜色抖动"],
            ["RandomResizedCrop", "scale=(0.8, 1.0)", "随机裁剪缩放"],
        ],
        col_widths=[4, 5, 5]
    )
    doc.add_paragraph(
        "注意：LoRA 训练数据不使用大幅增强——过度的几何变换可能破坏服饰的结构细节。"
    )

    # ============================================================
    # 五、图像分类模型
    # ============================================================
    doc.add_heading("五、图像分类模型", level=1)

    doc.add_heading("5.1 模型选型与训练策略", level=2)
    doc.add_paragraph(
        "采用 ResNet-50（torchvision 实现），加载 ImageNet 预训练权重进行迁移学习。"
        "ResNet-50 通过残差连接有效缓解深层网络的梯度消失问题，在中小规模数据集微调场景下表现稳定。"
    )
    doc.add_paragraph("训练分两阶段进行：")
    phases = [
        "阶段 1（分类头训练）：冻结 ResNet-50 主干网络，仅训练新替换的全连接分类头。学习率 1e-3，训练 10 个 epoch。",
        "阶段 2（全网络微调）：解冻所有层，以更低学习率 1e-4 对整个网络进行微调，训练 20 个 epoch。使用 Adam 优化器（weight_decay=1e-4）。",
    ]
    for p_text in phases:
        doc.add_paragraph(p_text, style="List Bullet")

    doc.add_paragraph(
        "学习率调度：ReduceLROnPlateau, mode='min', patience=2, factor=0.5——"
        "当验证 loss 连续 2 个 epoch 未下降时，学习率折半。"
        "早停机制：验证准确率连续 5 个 epoch 未提升则自动停止训练，防止过拟合。"
    )

    doc.add_heading("5.2 训练参数配置", level=2)
    add_styled_table(doc,
        ["参数", "推荐值", "说明"],
        [
            ["Batch Size", "32", "显存不足时可降至 16 或 8"],
            ["学习率（阶段1）", "1e-3", "只训练分类头，可使用较大学习率"],
            ["学习率（阶段2）", "1e-4", "解冻全网络，需降低学习率"],
            ["优化器", "Adam (weight_decay=1e-4)", "阶段2 使用权重衰减"],
            ["学习率调度", "ReduceLROnPlateau", "验证 loss 停止下降时自动降低学习率"],
            ["输入尺寸", "224×224×3 (RGB)", "ImageNet 标准尺寸"],
            ["输出类别", "3（藏族/羌族/彝族）", "对应 zang/qiang/yi"],
            ["模型大小", "~97MB (.pth)", "单文件，方便分发部署"],
        ],
        col_widths=[3.5, 5, 5.5]
    )

    doc.add_heading("5.3 模型评估与实际结果", level=2)
    doc.add_paragraph("评估指标：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1-Score，以及混淆矩阵可视化。")

    doc.add_paragraph("实际训练结果（测试集，总体准确率 88%）：")
    add_styled_table(doc,
        ["类别", "Precision", "Recall", "F1-Score", "说明"],
        [
            ["藏族 (zang)", "0.88", "0.92", "0.90", "召回率最高，藏族服饰特征辨识度较强"],
            ["羌族 (qiang)", "0.92", "0.85", "0.88", "精确率最高，误判为羌族的样本较少"],
            ["彝族 (yi)", "0.86", "0.88", "0.87", "表现均衡"],
            ["总体 (macro avg)", "0.88", "0.88", "0.88", "达到并超过 85% 目标"],
        ],
        col_widths=[3, 2.5, 2.5, 2.5, 3.5]
    )

    doc.add_paragraph(
        "推理性能：GPU 推理约 0.3 秒/张，CPU 推理约 1.5 秒/张。"
        "评估图表（混淆矩阵、ROC 曲线、训练曲线）输出至 outputs/figures/ 目录。"
    )

    # ============================================================
    # 六、LoRA 风格模型
    # ============================================================
    doc.add_heading("六、LoRA 风格模型", level=1)

    doc.add_heading("6.1 LoRA 原理简介", level=2)
    doc.add_paragraph(
        "LoRA（Low-Rank Adaptation，低秩适配）是一种参数高效微调（PEFT）技术。"
        "其核心思想是：冻结预训练大模型的原始权重，在旁路插入可训练的低秩矩阵（秩 r << 原始维度），"
        "仅更新这些低秩矩阵的参数。这样做的好处是："
    )
    benefits = [
        "训练参数量极少：23MB / 民族（相比原始 UNet 860MB），便于存储和分发",
        "训练成本低：150-170 张图、500 步、6GB 显存即可完成训练",
        "即插即用：推理时通过调整 lora_scale 控制风格强度（0.1~1.5），可随时切换不同民族的 LoRA",
    ]
    for b in benefits:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("6.2 训练数据准备", level=2)
    doc.add_paragraph("LoRA 训练数据的准备要求：")
    prep_items = [
        "图像尺寸：512×512（SD 1.5 标准分辨率），保持宽高比居中裁剪 + 黑边填充",
        "文本描述：每张图片配对一个同名 .txt 文件，内容为「触发词, 风格描述关键词」",
        "触发词规范：zang_costume / qiang_costume / yi_costume（全项目统一，便于模型学习稳定的风格映射）",
        "文件命名：<序号>_<触发词>.jpg（如 001_zang_costume.jpg），对应 001_zang_costume.txt",
    ]
    for item in prep_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_paragraph(
        "准备命令：python -m src.lora_training.prepare_dataset --src dataset/cleaned --out dataset/lora_data"
    )

    doc.add_heading("6.3 训练配置与参数", level=2)
    add_styled_table(doc,
        ["参数", "值", "说明"],
        [
            ["基础模型", "Stable Diffusion 1.5 (diffusers)", "本地加载，不依赖 WebUI"],
            ["LoRA Rank (r)", "16", "低秩矩阵维度，控制模型表达能力"],
            ["LoRA Alpha", "32", "缩放因子（实际 scale = alpha/r = 2.0）"],
            ["目标模块", "to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2", "作用于注意力层 + 前馈层"],
            ["Dropout", "0.05", "防止过拟合"],
            ["学习率", "1e-4", "余弦退火调度（cosine）"],
            ["训练步数", "500", "~3 epochs（150 张图, batch=1, grad_accum=4）"],
            ["Batch Size", "1", "低显存约束，通过梯度累积等效增大"],
            ["梯度累积步数", "4", "等效 batch_size = 1 × 4 = 4"],
            ["混合精度", "FP16", "减半显存占用"],
            ["输出格式", "safetensors", "安全、高效的模型权重格式"],
        ],
        col_widths=[4, 4.5, 5.5]
    )

    doc.add_heading("6.4 训练结果", level=2)
    doc.add_paragraph("本项目实际训练结果（参考：RTX 2060 6GB）：")
    add_styled_table(doc,
        ["民族", "数据量", "训练步数", "最终 Loss", "输出文件"],
        [
            ["藏族 (zang)", "167 张", "500", "~0.04", "zang_lora.safetensors (23MB)"],
            ["羌族 (qiang)", "170 张", "500", "~0.006", "qiang_lora.safetensors (23MB)"],
            ["彝族 (yi)", "160 张", "500", "~0.017", "yi_lora.safetensors (23MB)"],
        ],
        col_widths=[3, 2.5, 2.5, 2.5, 4.5]
    )
    doc.add_paragraph(
        "训练完成后删除大 checkpoint 文件（*_checkpoint_*.pt，每个约 3.5GB），只保留 .safetensors 文件。"
        "训练命令：python -m src.lora_training.train_lora --ethnicity all --steps 500"
    )

    # ============================================================
    # 七、Web 演示系统
    # ============================================================
    doc.add_heading("七、Web 演示系统", level=1)

    doc.add_heading("7.1 系统功能", level=2)
    doc.add_paragraph("Web 演示系统基于 FastAPI + Gradio 构建，启动后访问 http://localhost:8080 ，包含三个 Tab：")
    add_styled_table(doc,
        ["Tab", "功能", "说明"],
        [
            ["服饰识别", "上传图片 → ResNet-50 分类", "返回 Top-3 预测类别及置信度，含概率分布可视化"],
            ["AI 绘图", "选择民族 + 输入提示词 → SD+LoRA 生成", "支持 8 种采样器、自定义正/反向提示词、LoRA 强度调节"],
            ["系统信息", "技术栈和项目结构", "静态展示系统架构与技术说明"],
        ],
        col_widths=[3, 5, 6]
    )

    doc.add_paragraph("AI 绘图 Tab 的可调参数：")
    gen_params = [
        "民族选择：藏族 / 羌族 / 彝族（切换即加载对应 LoRA 权重）",
        "采样器：DPM++ 2M Karras / DPM++ SDE Karras / Euler a / Euler / DDIM / UniPC / PNDM / LMS（8 种可选）",
        "推理步数：10~50（默认 25），步数越多细节越丰富但耗时越长",
        "LoRA 强度：0.1~1.5（默认 0.8），控制民族风格的影响程度",
        "CFG 引导系数：1.0~15.0（默认 7.0），控制对提示词的遵循程度",
        "自定义正向提示词：覆盖预设模板，完全自定义生成方向",
        "自定义负向提示词：覆盖默认 EasyNegative，精确控制要避免的元素",
        "随机种子：-1（随机）或指定固定种子以复现结果",
    ]
    for p_text in gen_params:
        doc.add_paragraph(p_text, style="List Bullet")

    doc.add_heading("7.2 API 接口规格", level=2)

    doc.add_heading("GET /health — 健康检查", level=3)
    doc.add_paragraph(
        "返回系统状态，包括分类模型是否加载、可用 LoRA 模型列表、嵌入是否加载、SD 模型路径、当前设备等。"
    )

    doc.add_heading("POST /classify — 图像分类", level=3)
    doc.add_paragraph("请求：multipart/form-data，字段 file（图片文件，JPG/PNG/BMP/WebP，≤10MB）。")
    doc.add_paragraph(
        "图片预处理：Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet 均值/标准差)。"
    )
    doc.add_paragraph(
        "响应：{\"label\": \"藏族\", \"confidence\": 0.9234, \"top3\": [{\"label\": \"...\", \"confidence\": ...}, ...]}"
    )

    doc.add_heading("POST /generate — AI 文生图", level=3)
    doc.add_paragraph("请求：application/json，主要参数：")
    add_styled_table(doc,
        ["参数", "类型", "必填", "默认值", "范围", "说明"],
        [
            ["ethnicity", "string", "是", "-", "zang/qiang/yi", "目标民族"],
            ["prompt", "string", "否", "\"\"", "≤500字符", "补充描述"],
            ["steps", "int", "否", "25", "10~50", "推理步数"],
            ["lora_scale", "float", "否", "0.8", "0.1~1.5", "LoRA 权重"],
            ["cfg_scale", "float", "否", "7.0", "1.0~15.0", "CFG 引导系数"],
            ["seed", "int", "否", "-1", "-1或0~2³¹-1", "随机种子"],
            ["custom_positive", "string", "否", "\"\"", "≤500字符", "自定义正向提示词"],
            ["custom_negative", "string", "否", "\"\"", "≤500字符", "自定义负向提示词"],
            ["sampler", "string", "否", "DPM++ 2M Karras", "8种可选", "采样器"],
        ],
        col_widths=[2.5, 1.2, 1, 1.5, 2, 5.8]
    )
    doc.add_paragraph(
        "响应：{\"image_base64\": \"...\", \"positive_prompt\": \"...\", \"negative_prompt\": \"...\", \"parameters\": {...}}"
    )

    doc.add_heading("7.3 推理优化策略", level=2)
    doc.add_paragraph("为适配 3~6GB 低显存环境，系统默认启用以下优化：")
    optimizations = [
        "Attention Slicing：将注意力计算分片处理，减少峰值显存占用",
        "VAE Slicing：将 VAE 解码过程分块为 512×512，降低解码阶段显存需求",
        "FP16 推理：半精度浮点运算，显存占用减半",
        "safety_checker=None：跳过安全检查器，节省约 600MB 显存",
        "模型延迟加载：首次请求时才加载模型，避免启动时占用不必要的资源",
    ]
    for o in optimizations:
        doc.add_paragraph(o, style="List Bullet")

    doc.add_paragraph(
        "启动命令：python run_web.py --port 8080\n"
        "run_web.py 内置环境检测——若当前 Python 非 3.10 或无 CUDA，自动查找并切换到 costume310 conda 环境重新启动。"
    )

    # ============================================================
    # 八、部署流程
    # ============================================================
    doc.add_heading("八、部署流程", level=1)
    doc.add_paragraph(
        "完整部署流程共 12 步，详见 docs/deployment.md。以下为简要概览："
    )
    deploy_steps = [
        "创建 Conda 虚拟环境（Python 3.10）",
        "安装 PyTorch 2.4.0+cu124 及 torchvision 0.19.0+cu124",
        "解决 Windows DLL 依赖（vcomp140.dll + libomp140.x86_64.dll）",
        "安装项目依赖（pip install -r requirements.txt），处理版本冲突",
        "下载 Stable Diffusion 1.5 模型（ModelScope，约 4GB）",
        "放置 EasyNegative 文本反转嵌入（可选）",
        "数据采集与预处理（自动爬虫 → 清洗 → 划分）",
        "训练分类模型（两阶段迁移学习，目标准确率 ≥ 85%）",
        "准备 LoRA 训练数据（512×512 图像 + 触发词文本描述）",
        "训练 LoRA 模型（diffusers + peft，每类 500 步）",
        "启动 Web 演示系统（python run_web.py --port 8080）",
        "验证全系统（健康检查 + 分类测试 + 生成测试）",
    ]
    for i, s in enumerate(deploy_steps, 1):
        doc.add_paragraph(f"{i}. {s}")

    doc.add_paragraph(
        "一键全流程：python run.py（依次执行数据采集、预处理、训练、启动 Web）"
    )

    # ============================================================
    # 九、项目成果与交付
    # ============================================================
    doc.add_heading("九、项目成果与交付", level=1)
    add_styled_table(doc,
        ["交付成果", "要求", "状态", "说明"],
        [
            ["数据集", "三类服饰各 ≥200 张，含标注文件", "✓ 已完成", "dataset/annotations.csv + 三个民族图片"],
            ["识别模型", "测试集准确率 ≥85%，含混淆矩阵与分类报告", "✓ 88%", "models/classifier.pth"],
            ["LoRA 权重", "可在 Stable Diffusion 加载并生效", "✓ 已完成", "3 个民族，见 models/lora/"],
            ["Web 系统", "支持图像上传识别 + 文本输入生图", "✓ 已完成", "python run_web.py"],
            ["项目文档", "README + HELP + 部署文档 + 规格文档 + 本文件", "✓ 已完成", "见 docs/"],
        ],
        col_widths=[2.5, 4.5, 2, 5]
    )

    doc.add_paragraph(
        "系统特点总结：\n"
        "1) 全流程闭环：从原始数据采集到 Web 推理服务，覆盖机器学习项目的完整生命周期\n"
        "2) 本地离线运行：所有模型本地加载，不依赖任何外部 API，确保隐私和可用性\n"
        "3) 低显存优化：综合运用 attention slicing、VAE slicing、FP16 等技术，在 3~6GB 显存下可稳定运行\n"
        "4) 模块化设计：各功能模块独立，便于单独测试、替换和扩展\n"
        "5) 完善的文档体系：部署文档 + 使用帮助 + 技术规格 + 项目文档，覆盖不同读者的需求"
    )

    # ============================================================
    # 十、常见问题与故障排除
    # ============================================================
    doc.add_heading("十、常见问题与故障排除", level=1)
    add_styled_table(doc,
        ["症状", "原因", "解决方法"],
        [
            ["fbgemm.dll 加载失败", "缺少 LLVM/VC++ 运行时", "安装 VC++ Redist + 复制 libomp140.x86_64.dll"],
            ["No module named 'torchvision'", "安装 Gradio 时被卸载", "pip install torchvision-<版本>.whl"],
            ["torchvision 'no attribute extension'", "在非 Python 3.10 环境中运行", "用 conda activate costume310 激活后运行"],
            ["huggingface-hub 版本冲突", "Gradio 6.x 升级到 1.x", "pip install \"huggingface-hub<1.0,>=0.23.2\""],
            ["UNet 'expected 4 channels, got 3'", "训练未使用 VAE 编码", "检查 train_lora.py 是否正确加载 VAE"],
            ["CUDA out of memory", "显存不足", "降低 batch size / 分辨率，或使用 CPU"],
            ["Gradio \"Method not implemented\"", "后端函数报错", "检查服务器日志定位具体异常"],
            ["ModelScope 下载超时", "网络不稳定", "使用浏览器手动下载或 git clone 重试"],
            ["数据采集数量不足", "图片源不可达", "手动下载放入 dataset/raw/<民族>/"],
        ],
        col_widths=[4, 4, 6]
    )

    doc.add_paragraph(
        "更多问题请参阅 docs/HELP.md 和 docs/deployment.md。"
    )

    # ========== 保存 ==========
    doc.save(str(OUTPUT_PATH))
    print(f"文档已生成: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_docx()
