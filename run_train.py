#!/usr/bin/env python
"""
模型训练 — 独立入口
支持: ResNet-50 分类模型训练 + LoRA 风格模型训练
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="模型训练工具")
    parser.add_argument("--type", choices=["classifier", "lora", "all"], default="classifier",
                        help="训练类型: classifier=分类模型, lora=LoRA风格模型, all=全部")
    parser.add_argument("--data", type=str, default="dataset/split", help="分类训练数据目录")
    parser.add_argument("--lora-data", type=str, default="dataset/lora_data", help="LoRA训练数据目录")
    parser.add_argument("--batch", type=int, default=32, help="训练批次大小")
    parser.add_argument("--epochs1", type=int, default=10, help="阶段1训练轮数")
    parser.add_argument("--epochs2", type=int, default=20, help="阶段2训练轮数")
    parser.add_argument("--lora-steps", type=int, default=1500, help="LoRA训练步数")
    parser.add_argument("--lora-ethnicity", type=str, default="all",
                        choices=["zang", "qiang", "yi", "all"], help="LoRA训练的民族")
    args = parser.parse_args()

    print("=" * 50)
    print("模型训练流程")
    print("=" * 50)

    if args.type in ("classifier", "all"):
        print("\n" + "-" * 40)
        print("训练: ResNet-50 分类模型")
        print("-" * 40)

        # 检查数据就绪
        split_path = PROJECT_ROOT / args.data
        if not split_path.exists() or not list(split_path.glob("train/*")):
            print("错误: 训练数据未就绪，请先运行: python run_data.py")
            return

        from src.classification.train import train_model
        train_model(
            data_dir=str(split_path),
            model_save_path=str(PROJECT_ROOT / "models" / "classifier.pth"),
        )

        # 评估
        print("\n" + "-" * 40)
        print("评估: 分类模型")
        print("-" * 40)

        preds_path = PROJECT_ROOT / "outputs" / "logs" / "test_predictions.pt"
        if preds_path.exists():
            from src.classification.evaluate import run_evaluation
            run_evaluation(
                str(preds_path),
                str(PROJECT_ROOT / "models" / "classifier.pth"),
                str(PROJECT_ROOT / "outputs" / "figures"),
            )

    if args.type in ("lora", "all"):
        print("\n" + "-" * 40)
        print("训练: LoRA 风格模型")
        print("-" * 40)
        print("提示: LoRA 训练需要")
        print("  1. Stable Diffusion 基础模型（已配置本地路径 sd/）")
        print("  2. GPU（推荐 6GB+ 显存）")
        print("  3. 训练数据（可先运行 prepare_dataset.py）")

        # 检查 LoRA 数据
        lora_data_path = PROJECT_ROOT / args.lora_data
        if not lora_data_path.exists() or not list(lora_data_path.glob("*")):
            print("\nLoRA 训练数据未准备，正在准备...")
            from src.lora_training.prepare_dataset import prepare_dataset_for_lora
            prepare_dataset_for_lora(
                str(PROJECT_ROOT / "dataset" / "cleaned"),
                str(lora_data_path),
            )

        from src.lora_training.train_lora import train_lora, train_all
        if args.lora_ethnicity == "all":
            train_all(str(lora_data_path), str(PROJECT_ROOT / "models" / "lora"))
        else:
            train_lora(
                args.lora_ethnicity,
                str(lora_data_path),
                str(PROJECT_ROOT / "models" / "lora"),
            )

    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
