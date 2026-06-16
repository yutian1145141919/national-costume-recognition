"""
本地 SD + LoRA 推理模块
使用 diffusers 直接加载本地模型和 LoRA 权重进行文生图
"""
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "sd"
LORA_DIR = PROJECT_ROOT / "models" / "lora"
EMBEDDING_PATH = PROJECT_ROOT / "models" / "embeddings" / "EasyNegativeV2.safetensors"

ETHNICITY_LORA = {
    "zang": "zang_lora.safetensors",
    "qiang": "qiang_lora.safetensors",
    "yi": "yi_lora.safetensors",
}

# 全局状态
_pipe = None
_current_lora = None
_current_sampler = None

# 可选的采样器列表
SAMPLERS = [
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "Euler a",
    "Euler",
    "DDIM",
    "UniPC",
    "PNDM",
    "LMS",
]


def _load_pipeline():
    """延迟加载 SD 1.5 pipeline（不含 safety_checker，省显存）"""
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"加载 SD 1.5 模型到 {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    # 加载 EasyNegative 文本反转嵌入
    if EMBEDDING_PATH.exists():
        pipe.load_textual_inversion(str(EMBEDDING_PATH), token="easynegative")
        print("  EasyNegative 嵌入已加载")

    _pipe = pipe
    if device == "cuda":
        print(f"  模型加载完成，显存: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    else:
        print(f"  模型加载完成 (CPU)")
    return _pipe


def _apply_scheduler(pipe, sampler_name: str):
    """切换采样器/scheduler"""
    global _current_sampler
    if _current_sampler == sampler_name:
        return

    from diffusers import (
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        DDIMScheduler,
        UniPCMultistepScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
    )

    scheduler_cls = {
        "DPM++ 2M Karras": DPMSolverMultistepScheduler,
        "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "Euler": EulerDiscreteScheduler,
        "DDIM": DDIMScheduler,
        "UniPC": UniPCMultistepScheduler,
        "PNDM": PNDMScheduler,
        "LMS": LMSDiscreteScheduler,
    }.get(sampler_name)

    if scheduler_cls is None:
        print(f"  未知采样器: {sampler_name}，保持当前设置")
        return

    kwargs = {}
    if scheduler_cls is DPMSolverMultistepScheduler:
        kwargs["use_karras_sigmas"] = True
        kwargs["algorithm_type"] = "dpmsolver++"
    elif scheduler_cls is DPMSolverSinglestepScheduler:
        kwargs["use_karras_sigmas"] = True

    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
    _current_sampler = sampler_name
    print(f"  采样器: {sampler_name}")


def _apply_lora(pipe, ethnicity: str, lora_scale: float):
    """加载或切换 LoRA 权重并设置缩放比例"""
    global _peft_unet, _current_lora

    from safetensors.torch import load_file
    from peft import LoraConfig, get_peft_model

    lora_file = ETHNICITY_LORA.get(ethnicity)
    if lora_file is None:
        raise ValueError(f"未知民族: {ethnicity}")

    lora_path = LORA_DIR / lora_file
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA 权重不存在: {lora_path}")

    state_dict = load_file(str(lora_path))

    # 首次或切换民族时重新配置 LoRA
    if _current_lora != ethnicity:
        from peft import PeftModel
        # 如果 UNet 已被 peft 包装，先还原原始模型
        if isinstance(pipe.unet, PeftModel):
            pipe.unet = pipe.unet.merge_and_unload()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["to_q", "to_k", "to_v", "to_out.0",
                            "ff.net.0.proj", "ff.net.2"],
            bias="none",
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)
        _current_lora = ethnicity
        print(f"  LoRA [{ethnicity}] 已加载, missing={len(missing)}, unexpected={len(unexpected)}")

    # 动态调整 LoRA 缩放
    for module in pipe.unet.modules():
        if hasattr(module, "scaling") and "default" in module.scaling:
            module.scaling["default"] = lora_scale


def generate_local(
    ethnicity: str,
    user_prompt: str = "",
    lora_scale: float = 0.8,
    steps: int = 25,
    cfg_scale: float = 7.0,
    seed: int = -1,
    custom_positive: str = "",
    custom_negative: str = "",
    sampler: str = "DPM++ 2M Karras",
) -> tuple[Optional[Image.Image], str, str]:
    """
    使用本地 SD + LoRA 进行文生图

    参数:
        ethnicity: 民族标识 (zang/qiang/yi)
        user_prompt: 用户补充描述
        lora_scale: LoRA 权重 (0.1~1.5)
        steps: 推理步数
        cfg_scale: CFG 引导系数
        seed: 随机种子，-1 为随机
        custom_positive: 自定义正向提示词（为空则使用预设）
        custom_negative: 自定义负向提示词（为空则使用默认）
        sampler: 采样器名称
    返回:
        (image, positive_prompt, negative_prompt)
    """
    from .prompts import build_prompt

    positive, negative = build_prompt(ethnicity, user_prompt, lora_scale,
                                      custom_positive, custom_negative)

    try:
        pipe = _load_pipeline()
        _apply_lora(pipe, ethnicity, lora_scale)
        _apply_scheduler(pipe, sampler)
    except Exception as e:
        return None, f"模型加载失败: {e}", negative

    if seed < 0:
        seed = np.random.randint(0, 2**31 - 1)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=positive,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                width=512,
                height=512,
            )
        image = result.images[0]
        print(f"  生成完成, seed={seed}")
        return image, positive, negative
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, "显存不足，请降低分辨率或重启应用", negative
    except Exception as e:
        return None, f"推理失败: {e}", negative
