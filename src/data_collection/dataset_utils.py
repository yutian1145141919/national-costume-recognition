"""
数据集工具模块
提供数据集下载验证、格式检查、统计信息等功能
"""
import os
import csv
from pathlib import Path
from datetime import date

import pandas as pd
from PIL import Image


# 分类标签映射
LABEL_MAP = {
    "zang": 0,
    "qiang": 1,
    "yi": 2,
    0: "藏族",
    1: "羌族",
    2: "彝族",
}


def get_label_name(label) -> str:
    """将标签转换为中文名称"""
    if isinstance(label, str):
        return {"zang": "藏族", "qiang": "羌族", "yi": "彝族"}.get(label, label)
    return LABEL_MAP.get(label, str(label))


def get_label_id(label) -> int:
    """将标签转换为数字ID"""
    if isinstance(label, int):
        return label
    return {"zang": 0, "qiang": 1, "yi": 2}.get(label, -1)


def scan_dataset(base_dir: str) -> list[dict]:
    """
    扫描数据集目录，返回所有图片信息

    参数:
        base_dir: 数据集根目录（如 dataset/raw 或 dataset/cleaned）

    返回:
        包含 image_path, label, filename 的字典列表
    """
    base_path = Path(base_dir)
    records = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for ethnicity_dir in base_path.iterdir():
        if not ethnicity_dir.is_dir():
            continue
        label = ethnicity_dir.name
        for img_file in ethnicity_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                records.append({
                    "image_path": str(img_file.resolve()),
                    "label": label,
                    "filename": img_file.name,
                })

    return records


def generate_annotations_csv(
    raw_dir: str,
    output_path: str = "dataset/annotations.csv",
    annotator: str = "auto"
) -> pd.DataFrame:
    """
    根据目录结构生成分类标注CSV文件

    字段: image_path, label, source, quality_score, annotator, annotation_date

    参数:
        raw_dir: 原始图片目录
        output_path: 输出CSV路径
        annotator: 标注者名称

    返回:
        DataFrame
    """
    records = scan_dataset(raw_dir)
    today = date.today().isoformat()

    data = []
    for rec in records:
        data.append({
            "image_path": rec["image_path"],
            "label": rec["label"],
            "source": "web_crawl",
            "quality_score": 1,  # 初始默认可用
            "annotator": annotator,
            "annotation_date": today,
        })

    df = pd.DataFrame(data)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"标注文件已保存到 {output_path}（共 {len(df)} 条记录）")

    return df


def get_dataset_stats(base_dir: str) -> dict:
    """
    获取数据集统计信息

    参数:
        base_dir: 数据集目录

    返回:
        包含各类别图片数量、总计的字典
    """
    base_path = Path(base_dir)
    stats = {"total": 0, "categories": {}}

    for ethnicity_dir in base_path.iterdir():
        if ethnicity_dir.is_dir():
            count = len(list(ethnicity_dir.glob("*")))
            stats["categories"][ethnicity_dir.name] = count
            stats["total"] += count

    return stats


def validate_image(filepath: str) -> tuple[bool, str]:
    """
    验证图片文件是否有效

    参数:
        filepath: 图片文件路径

    返回:
        (是否有效, 错误信息)
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        # 再次打开以获取尺寸（verify后需要重新打开）
        with Image.open(filepath) as img:
            w, h = img.size
            if w < 50 or h < 50:
                return False, "图片尺寸过小"
        return True, ""
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "dataset" / "raw"

    print("数据集统计信息:")
    stats = get_dataset_stats(str(raw_dir))
    print(f"  总计: {stats['total']} 张")
    for cat, count in stats["categories"].items():
        label_name = get_label_name(cat)
        print(f"  {label_name}({cat}): {count} 张")

    # 生成标注文件
    csv_path = project_root / "dataset" / "annotations.csv"
    generate_annotations_csv(str(raw_dir), str(csv_path))
