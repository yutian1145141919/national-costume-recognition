"""
LoRA 训练配置模块
定义训练超参数、路径、触发词等核心配置
"""

# ============================================================
# 模型配置
# ============================================================

# Stable Diffusion 基础模型（可替换为本地路径）
BASE_MODEL_ID = "D:/pbl/民族lora/sd"  # 本地 Stable Diffusion 1.5 路径

# 输出目录
OUTPUT_DIR = "models/lora"
LORA_WEIGHTS_NAME = "ethnic_costume_lora.safetensors"

# ============================================================
# 三个民族的 LoRA 训练配置
# ============================================================

ETHNIC_CONFIGS = {
    "zang": {
        "name": "藏族",
        "trigger_word": "zang_costume",
        "instance_prompt": (
            "zang_costume, traditional Tibetan robe, colorful stripes, "
            "fur trim, silver jewelry, plateau background"
        ),
        "class_prompt": "traditional ethnic costume",
        "description_keywords": [
            "traditional Tibetan robe", "colorful stripes", "apron",
            "silver ornaments", "fur trim", "Lhasa background",
            "full body shot", "Tibetan plateau"
        ],
    },
    "qiang": {
        "name": "羌族",
        "trigger_word": "qiang_costume",
        "instance_prompt": (
            "qiang_costume, traditional Qiang dress, embroidered patterns, "
            "cloud shoes, sheepskin vest, mountain background"
        ),
        "class_prompt": "traditional ethnic costume",
        "description_keywords": [
            "traditional Qiang dress", "embroidery", "cloud shoes",
            "sheepskin vest", "silver headdress", "mountain background",
            "full body shot", "Sichuan region"
        ],
    },
    "yi": {
        "name": "彝族",
        "trigger_word": "yi_costume",
        "instance_prompt": (
            "yi_costume, traditional Yi dress, silver ornaments, "
            "pleated skirt, embroidered cape, Torch Festival"
        ),
        "class_prompt": "traditional ethnic costume",
        "description_keywords": [
            "traditional Yi dress", "silver jewelry", "pleated skirt",
            "embroidered cape", "earlva", "Torch Festival attire",
            "full body shot", "Yunnan region"
        ],
    },
}

# ============================================================
# 训练超参数
# ============================================================

TRAINING_CONFIG = {
    # 基础参数
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_train_steps": 1500,
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,

    # LoRA 参数
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
    ],

    # 正则化
    "with_prior_preservation": True,
    "prior_loss_weight": 1.0,
    "num_class_images": 100,

    # 保存与监控
    "save_steps": 250,
    "logging_steps": 25,
    "checkpointing_steps": 500,

    # 混合精度（Windows 建议 fp16=False 或使用 bf16）
    "mixed_precision": "no",  # "fp16", "bf16", "no"

    # 数据增强
    "center_crop": True,
    "random_flip": False,  # LoRA训练不建议翻转
    "color_jitter": False,
}

# ============================================================
# 推理配置
# ============================================================

INFERENCE_CONFIG = {
    "sampler": "DPM++ 2M Karras",
    "steps": 25,
    "cfg_scale": 7.0,
    "lora_scale": 0.8,
    "width": 512,
    "height": 512,
}

# ============================================================
# 默认提示词模板
# ============================================================

DEFAULT_POSITIVE_PROMPT = (
    "{trigger_word}, <lora:{lora_name}:{lora_scale}>, "
    "traditional ethnic costume, masterpiece, best quality, 8k, "
    "detailed fabric texture, intricate embroidery, natural lighting"
)

DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, blurry, deformed, ugly, worst quality, "
    "modern clothes, western style, watermark, text, signature"
)
