"""快速测试本地 SD + LoRA 推理"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from web.inference import _load_pipeline, _apply_lora
import torch

print("加载模型...")
pipe = _load_pipeline()

print("加载藏族 LoRA...")
_apply_lora(pipe, "zang", 0.8)

print("开始推理测试...")
positive = "zang_costume, traditional Tibetan robe, plateau background, masterpiece"
negative = "lowres, bad anatomy, blurry, deformed, worst quality"

with torch.inference_mode():
    result = pipe(
        prompt=positive,
        negative_prompt=negative,
        num_inference_steps=20,
        guidance_scale=7.0,
        width=512,
        height=512,
    )

img = result.images[0]
output_path = str(PROJECT_ROOT / "outputs" / "samples" / "test_zang_lora.png")
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
img.save(output_path)
print(f"推理成功! 图片已保存: {output_path}")
