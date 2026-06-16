"""
LoRA 训练数据准备模块
为每个民族准备图片+文本标注对，生成符合 diffusers 训练格式的数据集
"""
import os
import random
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from .config import ETHNIC_CONFIGS, TRAINING_CONFIG


def prepare_instance_images(
    source_dir: str,
    output_dir: str,
    ethnicity: str,
    target_size: int = 512,
    max_images: Optional[int] = None
) -> int:
    """
    准备实例图片（训练目标）

    参数:
        source_dir: 源图片目录
        output_dir: 输出目录
        ethnicity: 民族标识（zang/qiang/yi）
        target_size: 目标图片尺寸
        max_images: 最大图片数量

    返回:
        处理的图片数量
    """
    config = ETHNIC_CONFIGS[ethnicity]
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    source_path = Path(source_dir)

    # 收集图片
    images = [f for f in source_path.glob("*") if f.suffix.lower() in valid_extensions]
    if max_images:
        images = images[:max_images]

    print(f"  {ethnicity}: 准备 {len(images)} 张实例图片")

    for img_path in tqdm(images, desc=f"  处理 {ethnicity}"):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                # 居中裁剪为正方形
                w, h = img.size
                s = min(w, h)
                left = (w - s) // 2
                top = (h - s) // 2
                img = img.crop((left, top, left + s, top + s))
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

                out_path = os.path.join(output_dir, img_path.name.rsplit(".", 1)[0] + ".jpg")
                img.save(out_path, quality=95)

                # 生成对应的文本描述文件
                txt_path = out_path.rsplit(".", 1)[0] + ".txt"
                keywords = random.sample(
                    config["description_keywords"],
                    min(5, len(config["description_keywords"]))
                )
                caption = f"{config['trigger_word']}, {', '.join(keywords)}"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

        except Exception:
            continue

    return len(images)


def prepare_dataset_for_lora(
    source_dir: str,
    output_dir: str,
    target_size: int = 512
):
    """
    为所有民族准备 LoRA 训练数据

    参数:
        source_dir: 源图片根目录（含 zang/qiang/yi 子目录）
        output_dir: 输出目录
        target_size: 目标图片尺寸
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    for ethnicity, config in ETHNIC_CONFIGS.items():
        print(f"\n准备 {config['name']}({ethnicity}) 的 LoRA 训练数据...")

        instance_dir = output_path / ethnicity / "images"
        os.makedirs(instance_dir, exist_ok=True)

        # 搜索可能的源目录
        search_dirs = [
            source_path / ethnicity,
            source_path / f"{ethnicity}_lora512",
            source_path / f"{ethnicity}_cls224",
        ]

        found = None
        for sd in search_dirs:
            if sd.exists() and any(sd.glob("*")):
                found = sd
                break

        if found is None:
            print(f"  警告: 找不到 {ethnicity} 的源数据目录")
            continue

        n = prepare_instance_images(
            str(found), str(instance_dir), ethnicity, target_size
        )
        print(f"  处理完成: {n} 张图片 -> {instance_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA训练数据准备工具")
    parser.add_argument("--src", type=str, default="dataset/cleaned", help="源图片目录")
    parser.add_argument("--out", type=str, default="dataset/lora_data", help="输出目录")
    parser.add_argument("--size", type=int, default=512, help="目标尺寸")
    parser.add_argument("--ethnicity", type=str, choices=["zang", "qiang", "yi"],
                        help="只处理指定民族")
    parser.add_argument("--max", type=int, help="最大图片数量")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    if args.ethnicity:
        source_dir = project_root / args.src / args.ethnicity
        output_dir = project_root / args.out / args.ethnicity / "images"
        prepare_instance_images(str(source_dir), str(output_dir),
                                args.ethnicity, args.size, args.max)
    else:
        prepare_dataset_for_lora(
            str(project_root / args.src),
            str(project_root / args.out),
            args.size,
        )
