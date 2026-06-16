"""
图像分类推理模块
提供单张图片预测接口，返回类别与置信度
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


# ImageNet 归一化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["羌族", "彝族", "藏族"]  # 与 ImageFolder 字母序一致: qiang, yi, zang
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CostumeClassifier:
    """民族服饰分类器"""

    def __init__(self, model_path: str = "models/classifier.pth"):
        """
        初始化分类器

        参数:
            model_path: 训练好的模型权重文件路径
        """
        self.model_path = model_path
        self.device = DEVICE
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def _build_model(self) -> torch.nn.Module:
        """构建与训练时一致的模型结构"""
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features, NUM_CLASSES),
        )
        return model.to(self.device)

    def _load_weights(self):
        """加载训练权重"""
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"模型已从 {self.model_path} 加载到 {self.device}")

    def predict(self, image_path: str, top_k: int = 3) -> list[dict]:
        """
        对单张图片进行预测

        参数:
            image_path: 图片文件路径
            top_k: 返回前k个预测结果

        返回:
            [{"label": "藏族", "confidence": 0.92}, ...]
        """
        # 加载并预处理图片
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

        # 获取 top-k
        indices = probs.argsort()[::-1][:top_k]
        results = []
        for idx in indices:
            results.append({
                "label": CLASS_NAMES[idx],
                "confidence": round(float(probs[idx]), 4),
                "index": int(idx),
            })

        return results

    def predict_from_pil(self, img: Image.Image, top_k: int = 3) -> list[dict]:
        """从PIL Image对象预测"""
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

        indices = probs.argsort()[::-1][:top_k]
        results = []
        for idx in indices:
            results.append({
                "label": CLASS_NAMES[idx],
                "confidence": round(float(probs[idx]), 4),
                "index": int(idx),
            })

        return results


# 全局单例
_classifier_instance: CostumeClassifier | None = None


def get_classifier(model_path: str = "models/classifier.pth") -> CostumeClassifier:
    """获取全局分类器实例（延迟加载）"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = CostumeClassifier(model_path)
    return _classifier_instance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="民族服饰分类推理")
    parser.add_argument("image", type=str, help="图片文件路径")
    parser.add_argument("--model", type=str, default="models/classifier.pth", help="模型权重路径")
    parser.add_argument("--top", type=int, default=3, help="返回前k个结果")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    model_abs = project_root / args.model

    classifier = CostumeClassifier(str(model_abs))
    results = classifier.predict(args.image, top_k=args.top)

    print(f"\n图片: {args.image}")
    print("-" * 40)
    for i, r in enumerate(results):
        bar = "█" * int(r["confidence"] * 40)
        print(f"  {i + 1}. {r['label']}  {r['confidence']:.2%}  {bar}")
