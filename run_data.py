#!/usr/bin/env python
"""
数据采集与预处理 — 独立入口
执行流程: 爬虫下载 → 数据清洗 → 数据集划分 → 生成标注文件
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="数据集采集与预处理工具")
    parser.add_argument("--target", type=int, default=200, help="每类目标图片数量（默认200）")
    parser.add_argument("--source", choices=["bing", "baidu"], default="bing", help="图片来源")
    parser.add_argument("--skip-scrape", action="store_true", help="跳过数据采集，只执行预处理")
    parser.add_argument("--blur", type=float, default=100.0, help="模糊检测阈值")
    args = parser.parse_args()

    print("=" * 50)
    print("数据采集与预处理流程")
    print("=" * 50)

    if not args.skip_scrape:
        # 步骤1: 数据采集
        print("\n[1/4] 数据采集...")
        from src.data_collection.scraper import run_collection
        run_collection(
            base_dir=str(PROJECT_ROOT / "dataset" / "raw"),
            target_count=args.target,
        )

    # 步骤2: 数据清洗
    print("\n[2/4] 数据清洗...")
    from src.data_preprocessing.clean import run_cleaning_pipeline
    run_cleaning_pipeline(
        raw_dir=str(PROJECT_ROOT / "dataset" / "raw"),
        cleaned_dir=str(PROJECT_ROOT / "dataset" / "cleaned"),
        blur_threshold=args.blur,
    )

    # 步骤3: 合并分类数据并划分
    print("\n[3/4] 数据集划分...")
    import shutil, os
    cleaned_dir = PROJECT_ROOT / "dataset" / "cleaned"
    merged_dir = cleaned_dir / "merged_cls"

    # 清理旧合并目录
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    cls_dirs = list(cleaned_dir.glob("*_cls224"))
    if cls_dirs:
        for cls_dir in cls_dirs:
            ethnicity = cls_dir.name.replace("_cls224", "")
            dst = merged_dir / ethnicity
            os.makedirs(dst, exist_ok=True)
            for f in cls_dir.glob("*"):
                shutil.copy2(str(f), str(dst / f.name))

        from src.data_preprocessing.split import split_dataset
        split_dataset(str(merged_dir), str(PROJECT_ROOT / "dataset" / "split"))
    else:
        print("警告: 未找到清洗后的分类图片")
        print("请确保 raw/zang, raw/qiang, raw/yi 中有图片")

    # 步骤4: 生成标注文件
    print("\n[4/4] 生成标注文件...")
    from src.data_collection.dataset_utils import generate_annotations_csv
    generate_annotations_csv(
        str(PROJECT_ROOT / "dataset" / "raw"),
        str(PROJECT_ROOT / "dataset" / "annotations.csv"),
    )

    # 摘要
    print("\n" + "=" * 50)
    print("数据准备完成！")
    print("=" * 50)
    for split_name in ("train", "val", "test"):
        split_path = PROJECT_ROOT / "dataset" / "split" / split_name
        if split_path.exists():
            from src.data_collection.dataset_utils import get_dataset_stats
            stats = get_dataset_stats(str(split_path))
            print(f"  {split_name}: {stats['total']} 张")
    print(f"\n标注文件: dataset/annotations.csv")


if __name__ == "__main__":
    main()
