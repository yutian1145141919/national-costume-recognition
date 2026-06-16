"""
图像分类模型训练模块
基于 ResNet-50 迁移学习，实现三民族服饰分类
采用两阶段训练策略：冻结主干 → 解冻微调
"""
import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ============================================================
# 配置
# ============================================================

NUM_CLASSES = 3
CLASS_NAMES = ["羌族", "彝族", "藏族"]  # 与 ImageFolder 字母序一致: qiang, yi, zang
BATCH_SIZE = 32
NUM_EPOCHS_STAGE1 = 10  # 冻结主干阶段
NUM_EPOCHS_STAGE2 = 20  # 解冻全网络阶段
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet 归一化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool = True) -> transforms.Compose:
    """构建数据变换管道"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """创建 ResNet-50 模型（ImageNet 预训练）"""
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    # 替换全连接分类头
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


def freeze_backbone(model: nn.Module):
    """冻结主干网络，只保留分类头可训练"""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module):
    """解冻全部参数"""
    for param in model.parameters():
        param.requires_grad = True


def train_epoch(model, loader, criterion, optimizer, epoch: int, total_epochs: int) -> float:
    """训练一个epoch，返回平均loss"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"  训练 Epoch {epoch}/{total_epochs}")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    print(f"  Epoch {epoch}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
    return epoch_loss


def validate(model, loader, criterion) -> tuple[float, float, list, list]:
    """验证，返回(loss, accuracy, 预测列表, 真实标签列表)"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  验证中"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(
    data_dir: str = "dataset/split",
    model_save_path: str = "models/classifier.pth",
    log_dir: str = "outputs/logs"
):
    """
    执行完整训练流程

    参数:
        data_dir: 数据集划分目录（含 train/val/test 子目录）
        model_save_path: 模型保存路径
        log_dir: 日志输出目录
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"设备: {DEVICE}")
    print(f"数据目录: {data_dir}")

    # 加载数据集
    data_path = Path(data_dir)
    train_ds = datasets.ImageFolder(str(data_path / "train"), transform=build_transforms(True))
    val_ds = datasets.ImageFolder(str(data_path / "val"), transform=build_transforms(False))
    test_ds = datasets.ImageFolder(str(data_path / "test"), transform=build_transforms(False))

    print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张, 测试集: {len(test_ds)} 张")
    print(f"类别: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 构建模型
    model = create_model()
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------
    # 阶段1：冻结主干，只训练分类头
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("阶段1：冻结主干网络，训练分类头")
    print("=" * 50)

    freeze_backbone(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    patience_counter = 0
    history = {"stage1": [], "stage2": []}

    for epoch in range(1, NUM_EPOCHS_STAGE1 + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 epoch, NUM_EPOCHS_STAGE1)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["stage1"].append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc
        })
        print(f"  验证 Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> 保存最佳模型 (Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  早停触发，结束阶段1")
                break

    # ----------------------------------------------------------
    # 阶段2：解冻全部参数，低学习率微调
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("阶段2：解冻全网络，低学习率微调")
    print("=" * 50)

    # 加载阶段1最佳权重
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    unfreeze_all(model)
    optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS_STAGE2 + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 epoch, NUM_EPOCHS_STAGE2)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["stage2"].append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc
        })
        print(f"  验证 Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> 保存最佳模型 (Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  早停触发，结束阶段2")
                break

    # ----------------------------------------------------------
    # 最终测试
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("最终测试集评估")
    print("=" * 50)

    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion)
    print(f"测试集 Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

    # 保存训练历史
    history["test"] = {"loss": test_loss, "acc": test_acc}
    history["best_val_acc"] = best_val_acc
    history["timestamp"] = datetime.now().isoformat()

    history_path = os.path.join(log_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 保存测试预测结果供评估使用
    preds_path = os.path.join(log_dir, "test_predictions.pt")
    torch.save({
        "preds": test_preds,
        "labels": test_labels,
        "class_names": CLASS_NAMES,
    }, preds_path)

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型已保存到: {model_save_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="民族服饰分类模型训练")
    parser.add_argument("--data", type=str, default="dataset/split", help="数据集目录")
    parser.add_argument("--model", type=str, default="models/classifier.pth", help="模型保存路径")
    parser.add_argument("--batch", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs1", type=int, default=10, help="阶段1训练轮数")
    parser.add_argument("--epochs2", type=int, default=20, help="阶段2训练轮数")
    args = parser.parse_args()

    BATCH_SIZE = args.batch
    NUM_EPOCHS_STAGE1 = args.epochs1
    NUM_EPOCHS_STAGE2 = args.epochs2

    project_root = Path(__file__).resolve().parent.parent.parent
    data_abs = project_root / args.data
    model_abs = project_root / args.model

    train_model(str(data_abs), str(model_abs))
