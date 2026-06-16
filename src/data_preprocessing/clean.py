"""
数据清洗模块
功能：感知哈希去重、模糊检测过滤、统一尺寸调整
"""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm


def compute_phash(filepath: str) -> str:
    """计算图片的感知哈希值（用于去重）"""
    with Image.open(filepath) as img:
        return str(imagehash.phash(img))


def remove_duplicates(input_dir: str, threshold: int = 5) -> int:
    """
    使用 pHash 去除重复图片

    参数:
        input_dir: 图片目录
        threshold: 汉明距离阈值，小于此值视为重复

    返回:
        移除的图片数量
    """
    in_path = Path(input_dir)
    hashes: dict[str, str] = {}  # 文件路径 -> 哈希值
    to_remove: list[str] = []

    image_files = list(in_path.glob("*"))
    for f in tqdm(image_files, desc="  计算哈希"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            try:
                h = compute_phash(str(f))
                hashes[str(f)] = h
            except Exception:
                to_remove.append(str(f))  # 无法读取的直接标记

    # 两两比较找出重复
    hash_list = [(p, h) for p, h in hashes.items() if p not in to_remove]
    removed = set()

    for i in range(len(hash_list)):
        if hash_list[i][0] in removed:
            continue
        for j in range(i + 1, len(hash_list)):
            if hash_list[j][0] in removed:
                continue
            dist = sum(c1 != c2 for c1, c2 in zip(hash_list[i][1], hash_list[j][1]))
            if dist < threshold:
                # 保留第一个，移除第二个
                removed.add(hash_list[j][0])

    for f in removed:
        os.remove(f)
        to_remove.append(f)

    print(f"  去重完成：移除 {len(to_remove)} 张图片")
    return len(to_remove)


def detect_blur_laplacian(filepath: str, threshold: float = 100.0) -> tuple[bool, float]:
    """
    使用拉普拉斯方差检测模糊图片

    参数:
        filepath: 图片路径
        threshold: 方差阈值，低于此值判定为模糊

    返回:
        (是否模糊, 拉普拉斯方差值)
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


def remove_blurry(input_dir: str, threshold: float = 100.0) -> int:
    """
    移除模糊图片

    参数:
        input_dir: 图片目录
        threshold: 模糊阈值

    返回:
        移除的图片数量
    """
    in_path = Path(input_dir)
    removed = 0

    for f in tqdm(list(in_path.glob("*")), desc="  检测模糊"):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        is_blurry, var = detect_blur_laplacian(str(f), threshold)
        if is_blurry:
            os.remove(f)
            removed += 1

    print(f"  模糊检测完成：移除 {removed} 张图片")
    return removed


def resize_images(
    input_dir: str,
    output_dir: str,
    target_size: tuple[int, int] = (224, 224),
    keep_aspect: bool = False
) -> int:
    """
    统一调整图片尺寸

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        target_size: 目标尺寸 (宽, 高)
        keep_aspect: 是否保持宽高比（填充黑边）

    返回:
        处理的图片数量
    """
    os.makedirs(output_dir, exist_ok=True)
    in_path = Path(input_dir)
    count = 0

    for f in tqdm(list(in_path.glob("*")), desc="  调整尺寸"):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        try:
            with Image.open(f) as img:
                img = img.convert("RGB")
                if keep_aspect:
                    # 保持宽高比，用黑边填充
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    new_img = Image.new("RGB", target_size, (0, 0, 0))
                    paste_x = (target_size[0] - img.width) // 2
                    paste_y = (target_size[1] - img.height) // 2
                    new_img.paste(img, (paste_x, paste_y))
                else:
                    new_img = img.resize(target_size, Image.Resampling.LANCZOS)

                out_path = os.path.join(output_dir, f.name)
                new_img.save(out_path, quality=95)
                count += 1
        except Exception:
            continue

    print(f"  尺寸调整完成：处理 {count} 张图片 -> {target_size}")
    return count


def run_cleaning_pipeline(
    raw_dir: str,
    cleaned_dir: str,
    blur_threshold: float = 100.0,
    classifier_size: tuple[int, int] = (224, 224),
    lora_size: tuple[int, int] = (512, 512)
):
    """
    执行完整的数据清洗流程

    参数:
        raw_dir: 原始图片根目录（含 zang/qiang/yi 子目录）
        cleaned_dir: 清洗后输出目录
        blur_threshold: 模糊检测阈值
        classifier_size: 分类模型图片尺寸
        lora_size: LoRA训练图片尺寸
    """
    raw_path = Path(raw_dir)
    cleaned_path = Path(cleaned_dir)

    for ethnicity_dir in raw_path.iterdir():
        if not ethnicity_dir.is_dir():
            continue
        name = ethnicity_dir.name
        print(f"\n清洗 {name} 数据...")

        # 步骤1: 复制到临时目录
        tmp_dir = cleaned_path / f"{name}_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        for f in ethnicity_dir.glob("*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                shutil.copy2(str(f), str(tmp_dir / f.name))

        # 步骤2: 去重
        remove_duplicates(str(tmp_dir))

        # 步骤3: 去模糊
        remove_blurry(str(tmp_dir), blur_threshold)

        # 步骤4: 调整尺寸（分类用）
        cls_out = cleaned_path / f"{name}_cls224"
        resize_images(str(tmp_dir), str(cls_out), classifier_size)

        # 步骤5: 调整尺寸（LoRA用，保持宽高比）
        lora_out = cleaned_path / f"{name}_lora512"
        resize_images(str(tmp_dir), str(lora_out), lora_size, keep_aspect=True)

        # 清理临时目录
        shutil.rmtree(tmp_dir)
        print(f"  {name} 清洗完成")

    print(f"\n清洗流程全部完成，输出目录: {cleaned_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据清洗工具")
    parser.add_argument("--raw", type=str, default="dataset/raw", help="原始图片目录")
    parser.add_argument("--out", type=str, default="dataset/cleaned", help="输出目录")
    parser.add_argument("--blur", type=float, default=100.0, help="模糊检测阈值")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / args.raw
    cleaned_dir = project_root / args.out

    run_cleaning_pipeline(str(raw_dir), str(cleaned_dir), blur_threshold=args.blur)
