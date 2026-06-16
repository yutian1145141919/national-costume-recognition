"""
提示词模板模块
为三个民族提供文生图提示词模板
"""

# 默认负向提示词（所有民族通用）
# easynegative 为 EasyNegativeV2 文本反转嵌入的触发词
NEGATIVE_PROMPT = (
    "easynegative, lowres, bad anatomy, blurry, deformed, ugly, worst quality, "
    "modern clothes, western style, watermark, text, signature, "
    "distorted face, extra limbs, fused fingers, cropped"
)

# 推荐采样参数
DEFAULT_SAMPLER = "DPM++ 2M Karras"
DEFAULT_STEPS = 25
DEFAULT_CFG_SCALE = 7.0
DEFAULT_LORA_SCALE = 0.8


def build_prompt(ethnicity: str, user_description: str = "",
                 lora_scale: float = DEFAULT_LORA_SCALE,
                 custom_positive: str = "",
                 custom_negative: str = "") -> tuple[str, str]:
    """
    根据民族构建完整的正/负向提示词

    参数:
        ethnicity: 民族标识（zang/qiang/yi）
        user_description: 用户自定义描述（补充到预设提示词后，已废弃，改用 custom_positive）
        lora_scale: LoRA 权重
        custom_positive: 用户自定义正向提示词（为空则使用预设）
        custom_negative: 用户自定义负向提示词（为空则使用默认）

    返回:
        (positive_prompt, negative_prompt)
    """
    templates = {
        "zang": {
            "trigger": "zang_costume",
            "lora_name": "zang_lora",
            "base": (
                "zang_costume, traditional Tibetan robe, "
                "colorful stripes, fur trim, silver jewelry, "
                "plateau landscape background, "
                "masterpiece, best quality, 8k, detailed fabric texture"
            ),
        },
        "qiang": {
            "trigger": "qiang_costume",
            "lora_name": "qiang_lora",
            "base": (
                "qiang_costume, traditional Qiang ethnic dress, "
                "intricate embroidery, cloud pattern shoes, "
                "sheepskin vest, silver headdress, "
                "mountain village background, "
                "masterpiece, best quality, 8k, detailed embroidery"
            ),
        },
        "yi": {
            "trigger": "yi_costume",
            "lora_name": "yi_lora",
            "base": (
                "yi_costume, traditional Yi ethnic dress, "
                "silver ornaments, pleated skirt, "
                "embroidered cape, earring decorations, "
                "Torch Festival atmosphere, "
                "masterpiece, best quality, 8k, detailed silverwork"
            ),
        },
    }

    template = templates.get(ethnicity)
    if template is None:
        raise ValueError(f"未知的民族标识: {ethnicity}，支持: zang, qiang, yi")

    # 正向提示词：优先使用自定义，否则使用预设
    if custom_positive.strip():
        positive = custom_positive.strip()
    else:
        positive = template["base"]
        if user_description:
            positive = f"{positive}, {user_description}"

    # 负向提示词：优先使用自定义，否则使用默认
    negative = custom_negative.strip() if custom_negative.strip() else NEGATIVE_PROMPT

    return positive, negative


def get_prompt_guide() -> dict:
    """获取各民族的提示词指南"""
    return {
        "zang": {
            "name": "藏族",
            "trigger_word": "zang_costume",
            "style_keywords": [
                "藏袍 (Tibetan robe)", "彩色条纹 (colorful stripes)",
                "银饰 (silver jewelry)", "毛皮装饰 (fur trim)",
                "围裙 (apron)", "高原背景 (plateau background)",
            ],
            "example_prompt": (
                "zang_costume, traditional Tibetan robe, colorful stripes, "
                "silver ornaments, plateau landscape, masterpiece, 8k"
            ),
        },
        "qiang": {
            "name": "羌族",
            "trigger_word": "qiang_costume",
            "style_keywords": [
                "刺绣 (embroidery)", "云云鞋 (cloud shoes)",
                "羊皮褂 (sheepskin vest)", "银头饰 (silver headdress)",
                "绣花围腰 (embroidered apron)", "山地背景 (mountain background)",
            ],
            "example_prompt": (
                "qiang_costume, intricate embroidery, cloud shoes, "
                "silver headdress, mountain village, masterpiece, 8k"
            ),
        },
        "yi": {
            "name": "彝族",
            "trigger_word": "yi_costume",
            "style_keywords": [
                "银饰 (silver ornaments)", "百褶裙 (pleated skirt)",
                "擦尔瓦 (felt cape)", "刺绣服饰 (embroidered clothing)",
                "火把节 (Torch Festival)", "滇川背景 (Yunnan-Sichuan)",
            ],
            "example_prompt": (
                "yi_costume, silver jewelry, pleated skirt, "
                "embroidered cape, Torch Festival scene, masterpiece, 8k"
            ),
        },
    }
