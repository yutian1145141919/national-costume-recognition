"""
模型评估模块
功能：混淆矩阵、分类报告（精确率/召回率/F1）、ROC曲线
"""
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，适配无桌面环境
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(y_true: list[int], y_pred: list[int],
                          class_names: list[str], save_path: str = "confusion_matrix.png"):
    """绘制并保存混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title("混淆矩阵", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")
    return cm


def plot_roc_curves(y_true: list[int], y_scores: list[list[float]],
                    class_names: list[str], save_path: str = "roc_curves.png"):
    """
    绘制多分类ROC曲线（One-vs-Rest）

    参数:
        y_true: 真实标签列表
        y_scores: 每个样本各类别的预测概率（n_samples × n_classes）
        class_names: 类别名称
        save_path: 保存路径
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(y_scores)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("假正率 (False Positive Rate)", fontsize=12)
    plt.ylabel("真正率 (True Positive Rate)", fontsize=12)
    plt.title("ROC 曲线 (One-vs-Rest)", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC曲线已保存到: {save_path}")


def plot_training_history(history_path: str, save_dir: str):
    """绘制训练历史曲线"""
    import json
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax = axes[0]
    for stage_name, color in [("stage1", "blue"), ("stage2", "orange")]:
        if stage_name in history:
            epochs = [h["epoch"] for h in history[stage_name]]
            losses = [h["train_loss"] for h in history[stage_name]]
            val_losses = [h["val_loss"] for h in history[stage_name]]
            offset = 0 if stage_name == "stage1" else len(history.get("stage1", []))
            x = [e + offset for e in epochs]
            ax.plot(x, losses, color=color, linestyle="-", alpha=0.5, label=f"{stage_name} 训练")
            ax.plot(x, val_losses, color=color, linestyle="--", label=f"{stage_name} 验证")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练 Loss 曲线")
    ax.legend()

    # Accuracy 曲线
    ax = axes[1]
    for stage_name, color in [("stage1", "blue"), ("stage2", "orange")]:
        if stage_name in history:
            epochs = [h["epoch"] for h in history[stage_name]]
            accs = [h["val_acc"] for h in history[stage_name]]
            offset = 0 if stage_name == "stage1" else len(history.get("stage1", []))
            x = [e + offset for e in epochs]
            ax.plot(x, accs, color=color, linestyle="-", label=f"{stage_name} 验证准确率")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("验证准确率曲线")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


def run_evaluation(
    predictions_path: str,
    model_path: str,
    output_dir: str = "outputs/figures",
    class_names: list[str] | None = None
):
    """
    执行完整的模型评估

    参数:
        predictions_path: 测试集预测结果文件路径（.pt）
        model_path: 模型权重文件路径
        output_dir: 图表输出目录
        class_names: 类别名称列表
    """
    if class_names is None:
        class_names = ["羌族", "彝族", "藏族"]  # 与 ImageFolder 字母序一致: qiang, yi, zang

    os.makedirs(output_dir, exist_ok=True)

    # 加载预测结果
    data = torch.load(predictions_path, weights_only=False)
    y_pred = data["preds"] if "preds" in data else data["predictions"]
    y_true = data["labels"] if "labels" in data else data["labels_true"]

    print("=" * 50)
    print("模型评估报告")
    print("=" * 50)

    # 1. 整体准确率
    acc = accuracy_score(y_true, y_pred)
    print(f"\n整体准确率 (Accuracy): {acc:.4f} ({acc * 100:.2f}%)")
    print(f"是否达标 (>=85%): {'[是]' if acc >= 0.85 else '[否]'}")

    # 2. 分类报告
    print("\n分类报告:")
    print("-" * 60)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(report)

    # 3. 各类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names)))
    )
    print("-" * 60)
    print(f"{'类别':<8} {'精确率':>8} {'召回率':>8} {'F1-Score':>10} {'样本数':>8}")
    for i, name in enumerate(class_names):
        print(f"{name:<8} {precision[i]:>8.4f} {recall[i]:>8.4f} {f1[i]:>10.4f} {support[i]:>8}")

    # 4. 混淆矩阵
    cm = plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    # 5. 训练历史曲线
    history_path = os.path.join(os.path.dirname(output_dir), "logs", "training_history.json")
    if os.path.exists(history_path):
        plot_training_history(history_path, output_dir)

    # 6. 生成文本报告
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("民族服饰识别模型评估报告\n")
        f.write(f"整体准确率: {acc:.4f} ({acc * 100:.2f}%)\n")
        f.write(f"达标情况: {'是' if acc >= 0.85 else '否'}\n\n")
        f.write("分类报告:\n")
        f.write(report)
        f.write("\n\n混淆矩阵:\n")
        f.write(str(cm))
    print(f"\n评估报告已保存到: {report_path}")

    return acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型评估工具")
    parser.add_argument("--preds", type=str, default="outputs/logs/test_predictions.pt",
                        help="测试预测结果文件")
    parser.add_argument("--model", type=str, default="models/classifier.pth",
                        help="模型权重文件")
    parser.add_argument("--out", type=str, default="outputs/figures", help="输出目录")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    preds_abs = project_root / args.preds
    output_abs = project_root / args.out

    run_evaluation(str(preds_abs), str(project_root / args.model), str(output_abs))
