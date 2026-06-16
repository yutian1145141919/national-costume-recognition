#!/usr/bin/env python
"""
民族服饰识别与AI绘图系统 — 主入口
一键执行完整流程：数据采集 → 预处理 → 训练 → Web服务

使用方法:
    python run.py              # 执行完整流程
    python run.py --data-only  # 只执行数据采集+预处理
    python run.py --train-only # 只执行模型训练
    python run.py --web-only   # 只启动Web服务
"""
import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_data_collection(target: int = 200):
    """步骤1: 数据采集"""
    print("\n" + "=" * 60)
    print("步骤 1/4: 数据采集")
    print("=" * 60)
    from src.data_collection.scraper import run_collection
    run_collection(base_dir=str(PROJECT_ROOT / "dataset" / "raw"), target_count=target)


def step_data_preprocessing():
    """步骤2: 数据预处理"""
    print("\n" + "=" * 60)
    print("步骤 2/4: 数据预处理")
    print("=" * 60)

    raw_dir = str(PROJECT_ROOT / "dataset" / "raw")
    cleaned_dir = str(PROJECT_ROOT / "dataset" / "cleaned")

    # 2a: 数据清洗
    print("\n--- 数据清洗 ---")
    from src.data_preprocessing.clean import run_cleaning_pipeline
    run_cleaning_pipeline(raw_dir, cleaned_dir)

    # 2b: 数据集划分
    print("\n--- 数据集划分 ---")
    # 查找清洗后的分类数据集目录
    cls_dirs = list(Path(cleaned_dir).glob("*_cls224"))
    if cls_dirs:
        # 合并各类别到统一目录
        merged_dir = PROJECT_ROOT / "dataset" / "cleaned" / "merged_cls"
        import shutil, os
        for cls_dir in cls_dirs:
            ethnicity = cls_dir.name.replace("_cls224", "")
            dst = merged_dir / ethnicity
            os.makedirs(dst, exist_ok=True)
            for f in cls_dir.glob("*"):
                shutil.copy2(str(f), str(dst / f.name))

        from src.data_preprocessing.split import split_dataset
        split_dataset(str(merged_dir), str(PROJECT_ROOT / "dataset" / "split"))
    else:
        print("警告: 未找到清洗后的分类图片，跳过数据集划分")

    # 2c: 生成标注文件
    print("\n--- 生成标注文件 ---")
    from src.data_collection.dataset_utils import generate_annotations_csv
    generate_annotations_csv(raw_dir, str(PROJECT_ROOT / "dataset" / "annotations.csv"))


def step_training():
    """步骤3: 模型训练"""
    print("\n" + "=" * 60)
    print("步骤 3/4: 模型训练")
    print("=" * 60)

    # 3a: 分类模型训练
    print("\n--- 分类模型训练 ---")
    from src.classification.train import train_model
    train_model(
        data_dir=str(PROJECT_ROOT / "dataset" / "split"),
        model_save_path=str(PROJECT_ROOT / "models" / "classifier.pth"),
    )

    # 3b: 模型评估
    print("\n--- 模型评估 ---")
    from src.classification.evaluate import run_evaluation
    preds_path = PROJECT_ROOT / "outputs" / "logs" / "test_predictions.pt"
    if preds_path.exists():
        run_evaluation(str(preds_path), str(PROJECT_ROOT / "models" / "classifier.pth"),
                       str(PROJECT_ROOT / "outputs" / "figures"))
    else:
        print("警告: 未找到测试预测结果，跳过评估")

    # 3c: LoRA 训练（需要 GPU）
    print("\n--- LoRA 训练 ---")
    print("提示: LoRA 训练需要 Stable Diffusion 基础模型（已配置本地路径 sd/）和 GPU")
    print("若要训练 LoRA，请运行:")
    print("  python -m src.lora_training.train_lora --ethnicity all")


def step_web_service():
    """步骤4: 启动 Web 服务"""
    print("\n" + "=" * 60)
    print("步骤 4/4: 启动 Web 演示服务")
    print("=" * 60)

    # 检查模型
    classifier_path = PROJECT_ROOT / "models" / "classifier.pth"
    if classifier_path.exists():
        print(f"分类模型: 已就绪 ({classifier_path})")
    else:
        print("分类模型: 未找到，服饰识别功能将不可用")
        print("  请先运行: python run.py --train-only")

    lora_dir = PROJECT_ROOT / "models" / "lora"
    lora_files = list(lora_dir.glob("*_lora.*"))
    if lora_files:
        print(f"LoRA 模型: {len(lora_files)} 个已就绪")
    else:
        print("LoRA 模型: 未找到，AI 绘图功能需要额外配置")

    from src.web.app import launch_app
    launch_app(server_port=8080)


def main():
    parser = argparse.ArgumentParser(
        description="民族服饰识别与AI绘图系统 — 一键执行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                  执行完整流程
  python run.py --data-only      只执行数据采集与预处理
  python run.py --train-only     只执行模型训练
  python run.py --web-only       只启动Web服务
        """,
    )
    parser.add_argument("--data-only", action="store_true", help="只执行数据采集与预处理")
    parser.add_argument("--train-only", action="store_true", help="只执行模型训练")
    parser.add_argument("--web-only", action="store_true", help="只启动Web服务")
    parser.add_argument("--target", type=int, default=200, help="每类目标图片数量")
    args = parser.parse_args()

    print("=" * 60)
    print("民族服饰识别与AI绘图综合实践系统")
    print("Ethnic Costume Recognition & AI Drawing System")
    print("=" * 60)

    has_specific = args.data_only or args.train_only or args.web_only

    if args.data_only:
        step_data_collection(args.target)
        step_data_preprocessing()
    elif args.train_only:
        step_training()
    elif args.web_only:
        step_web_service()
    else:
        # 完整流程
        step_data_collection(args.target)
        step_data_preprocessing()
        step_training()
        step_web_service()

    print("\n" + "=" * 60)
    print("执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
