"""
数据集划分模块
按 70/15/15 比例划分训练集、验证集、测试集
"""
import os
import shutil
import random
from pathlib import Path

from tqdm import tqdm


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> dict[str, dict[str, int]]:
    """
    将数据集按比例划分为 train/val/test

    参数:
        input_dir: 输入目录（包含各类别子目录）
        output_dir: 输出目录（将创建 train/val/test 子目录）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    返回:
        各集合各类别的图片数量统计
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "比例之和必须为1"

    random.seed(seed)
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # 创建输出目录结构
    for split_name in ("train", "val", "test"):
        for category in in_path.iterdir():
            if category.is_dir():
                os.makedirs(out_path / split_name / category.name, exist_ok=True)

    stats = {split: {} for split in ("train", "val", "test")}
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for category_dir in sorted(in_path.iterdir()):
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name

        # 收集所有图片文件
        images = []
        for f in category_dir.glob("*"):
            if f.suffix.lower() in valid_extensions:
                images.append(f)

        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # 剩余的归入测试集

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # 复制文件
        print(f"\n划分 {category_name}（共 {n_total} 张）:")
        for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            dst_dir = out_path / split_name / category_name
            for img in tqdm(split_images, desc=f"  -> {split_name}"):
                shutil.copy2(str(img), str(dst_dir / img.name))
            stats[split_name][category_name] = len(split_images)
            print(f"  {split_name}: {len(split_images)} 张")

    # 打印汇总
    print(f"\n{'='*40}")
    print("数据集划分汇总:")
    print(f"{'类别':<12} {'训练集':>8} {'验证集':>8} {'测试集':>8} {'总计':>8}")
    print("-" * 40)
    total_train = total_val = total_test = 0
    for cat in sorted(stats["train"].keys()):
        tr = stats["train"].get(cat, 0)
        va = stats["val"].get(cat, 0)
        te = stats["test"].get(cat, 0)
        print(f"{cat:<12} {tr:>8} {va:>8} {te:>8} {tr + va + te:>8}")
        total_train += tr
        total_val += va
        total_test += te
    print("-" * 40)
    print(f"{'合计':<12} {total_train:>8} {total_val:>8} {total_test:>8} {total_train + total_val + total_test:>8}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument("--input", type=str, required=True, help="输入目录（含类别子目录）")
    parser.add_argument("--out", type=str, default="dataset/split", help="输出目录")
    parser.add_argument("--train", type=float, default=0.70, help="训练集比例")
    parser.add_argument("--val", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    input_abs = project_root / args.input
    output_abs = project_root / args.out

    split_dataset(str(input_abs), str(output_abs),
                  train_ratio=args.train, val_ratio=args.val,
                  test_ratio=args.test, seed=args.seed)
