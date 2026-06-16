"""
数据增强模块
为分类模型训练提供在线/离线数据增强
"""
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def random_horizontal_flip(img: Image.Image, p: float = 0.5) -> Image.Image:
    """随机水平翻转"""
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_rotation(img: Image.Image, max_degrees: float = 15.0) -> Image.Image:
    """随机旋转 ±max_degrees"""
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)


def random_color_jitter(img: Image.Image, brightness: float = 0.2,
                        contrast: float = 0.2, saturation: float = 0.2) -> Image.Image:
    """
    随机颜色抖动
    使用 PIL ImageEnhance 实现
    """
    from PIL import ImageEnhance

    # 亮度
    factor = random.uniform(1 - brightness, 1 + brightness)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 对比度
    factor = random.uniform(1 - contrast, 1 + contrast)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 饱和度
    factor = random.uniform(1 - saturation, 1 + saturation)
    img = ImageEnhance.Color(img).enhance(factor)

    return img


def random_resized_crop(img: Image.Image, target_size: tuple[int, int] = (224, 224),
                        scale: tuple[float, float] = (0.8, 1.0),
                        ratio: tuple[float, float] = (0.75, 1.33)) -> Image.Image:
    """
    随机裁剪并缩放
    模拟 torchvision.transforms.RandomResizedCrop
    """
    w, h = img.size
    area = w * h

    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])

        crop_w = int(np.sqrt(target_area * aspect_ratio))
        crop_h = int(np.sqrt(target_area / aspect_ratio))

        if crop_w <= w and crop_h <= h:
            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
            cropped = img.crop((x, y, x + crop_w, y + crop_h))
            return cropped.resize(target_size, Image.Resampling.LANCZOS)

    # 兜底：中心裁剪
    return img.resize(target_size, Image.Resampling.LANCZOS)


def augment_single_image(img_path: str, output_dir: str, n_augment: int = 3,
                         target_size: tuple[int, int] = (224, 224)):
    """
    对单张图片执行数据增强，生成多个变体

    参数:
        img_path: 原始图片路径
        output_dir: 输出目录
        n_augment: 每张图片生成的增强变体数量
        target_size: 目标尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    with Image.open(img_path) as img:
        img = img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)

        for i in range(n_augment):
            aug = img.copy()
            aug = random_horizontal_flip(aug)
            aug = random_rotation(aug, max_degrees=15)
            aug = random_color_jitter(aug)
            aug = random_resized_crop(aug, target_size)

            out_path = os.path.join(output_dir, f"{base_name}_aug{i}.jpg")
            aug.save(out_path, quality=95)


def augment_dataset(input_dir: str, output_dir: str, n_augment: int = 3,
                    target_size: tuple[int, int] = (224, 224),
                    categories: list[str] | None = None):
    """
    对整个数据集执行离线数据增强

    参数:
        input_dir: 输入目录（含类别子目录）
        output_dir: 输出目录
        n_augment: 每张图片的增强倍数
        target_size: 目标尺寸
        categories: 要增强的类别列表，None表示所有类别
    """
    in_path = Path(input_dir)

    for category_dir in in_path.iterdir():
        if not category_dir.is_dir():
            continue
        if categories and category_dir.name not in categories:
            continue

        cat_out = os.path.join(output_dir, category_dir.name)
        os.makedirs(cat_out, exist_ok=True)

        images = [f for f in category_dir.glob("*")
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]

        # 先复制原始图片
        for img in images:
            import shutil
            dst = os.path.join(cat_out, img.name)
            if not os.path.exists(dst):
                shutil.copy2(str(img), dst)

        # 生成增强变体
        for img in tqdm(images, desc=f"增强 {category_dir.name}"):
            augment_single_image(str(img), cat_out, n_augment, target_size)

        print(f"  {category_dir.name}: {len(images)} 张原始 -> "
              f"约 {len(images) * (n_augment + 1)} 张（含增强）")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据增强工具")
    parser.add_argument("--input", type=str, required=True, help="输入目录")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--n", type=int, default=3, help="每张图片增强倍数")
    parser.add_argument("--size", type=int, default=224, help="目标尺寸")
    args = parser.parse_args()

    augment_dataset(args.input, args.out, n_augment=args.n,
                    target_size=(args.size, args.size))
