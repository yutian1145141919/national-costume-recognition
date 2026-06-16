"""
LoRA 微调训练模块
基于 diffusers + peft 库实现民族服饰风格 LoRA 权重训练
支持 Windows 环境，可在有限显存下运行
"""
import os
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from .config import (
    ETHNIC_CONFIGS, TRAINING_CONFIG, BASE_MODEL_ID, OUTPUT_DIR
)


class LoraDataset(Dataset):
    """LoRA 训练数据集：图片+文本标注对"""

    def __init__(self, image_dir: str, tokenizer, size: int = 512):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.size = size

        self.image_paths = sorted(
            p for p in self.image_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 加载图片
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)
        # 转为 [-1, 1] 范围
        img_np = list(img.getdata())
        img_tensor = torch.tensor(img_np, dtype=torch.float32
        ).reshape(self.size, self.size, 3) / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1)

        # 加载文本标注
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = "traditional ethnic costume"

        return {
            "pixel_values": img_tensor,
            "prompt": prompt,
            "image_path": str(img_path),
        }


def collate_fn(batch, tokenizer, max_length: int = 77):
    """批次整理函数"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    prompts = [item["prompt"] for item in batch]

    # Tokenize
    tokens = tokenizer(
        prompts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
    }


def train_lora(
    ethnicity: str,
    data_dir: str = "dataset/lora_data",
    output_dir: str = "models/lora",
    resume_from_checkpoint: Optional[str] = None
):
    """
    训练单个民族的 LoRA 权重

    参数:
        ethnicity: 民族标识（zang/qiang/yi）
        data_dir: 训练数据根目录
        output_dir: 模型输出目录
        resume_from_checkpoint: 从检查点恢复训练
    """
    config = ETHNIC_CONFIGS[ethnicity]
    train_cfg = TRAINING_CONFIG

    print(f"\n{'='*60}")
    print(f"训练 {config['name']}({ethnicity}) LoRA 模型")
    print(f"{'='*60}")

    # 尝试导入 diffusers 和 peft
    try:
        from diffusers import (
            DDPMScheduler,
            UNet2DConditionModel,
            AutoencoderKL,
        )
        from transformers import CLIPTokenizer, CLIPTextModel
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行: pip install diffusers peft transformers accelerate")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    if device.type != "cuda":
        print("警告: LoRA 训练建议使用 GPU，CPU 训练将极其缓慢")

    # 数据路径
    image_dir = Path(data_dir) / ethnicity / "images"
    if not image_dir.exists():
        print(f"错误: 数据目录不存在: {image_dir}")
        print("请先运行 prepare_dataset.py 准备训练数据")
        return None

    # 加载模型组件
    print(f"加载基础模型: {BASE_MODEL_ID}")

    try:
        tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("可以使用 HuggingFace 镜像或本地模型路径")
        return None

    # 冻结所有参数
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 配置 LoRA
    lora_config = LoraConfig(
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none",
    )

    # 将 LoRA 应用于 UNet
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"LoRA 可训练参数: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # 移动到设备
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)

    # 创建数据集
    dataset = LoraDataset(str(image_dir), tokenizer, train_cfg["resolution"])
    dataloader = DataLoader(
        dataset, batch_size=train_cfg["train_batch_size"],
        shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    print(f"数据集大小: {len(dataset)} 张图片")
    print(f"训练步数: {train_cfg['max_train_steps']}")

    # 优化器
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=1e-4,
    )

    # 学习率调度器（余弦退火）
    lr_lambda = lambda step: max(
        0.0,
        0.5 * (1.0 + math.cos(math.pi * step / train_cfg["max_train_steps"]))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    running_loss = 0.0
    best_loss = float("inf")
    start_epoch = 0

    # 恢复检查点
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, weights_only=False)
        unet.load_state_dict(checkpoint["unet_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint.get("epoch", 0)
        print(f"从检查点恢复: 步数 {global_step}")

    num_epochs = max(1, train_cfg["max_train_steps"] // len(dataloader) + 1)

    print("\n开始训练...")
    print("-" * 60)

    for epoch in range(start_epoch, num_epochs):
        if global_step >= train_cfg["max_train_steps"]:
            break

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            if global_step >= train_cfg["max_train_steps"]:
                break

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # 编码文本 + VAE latent
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # 0.18215

            # 对 latent 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet 预测噪声
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 计算损失
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            # 梯度累积
            if (global_step + 1) % train_cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "step": global_step,
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # 日志记录
            if global_step % train_cfg["logging_steps"] == 0:
                avg_loss = running_loss / train_cfg["logging_steps"]
                print(f"\n  步数 {global_step}: 平均Loss = {avg_loss:.4f}")
                running_loss = 0.0

            # 保存检查点
            if global_step % train_cfg["checkpointing_steps"] == 0:
                checkpoint_path = os.path.join(
                    output_dir, f"{ethnicity}_checkpoint_{global_step}.pt"
                )
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch,
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, checkpoint_path)
                print(f"\n  检查点已保存: {checkpoint_path}")

    # 保存最终 LoRA 权重
    final_path = os.path.join(output_dir, f"{ethnicity}_lora.safetensors")

    # 提取 LoRA 权重（只保存 adapter 部分）
    try:
        from safetensors.torch import save_file
        lora_state_dict = {
            k: v for k, v in unet.state_dict().items()
            if "lora" in k.lower()
        }
        save_file(lora_state_dict, final_path)
        print(f"\nLoRA 权重已保存: {final_path}")
        print(f"LoRA 参数量: {len(lora_state_dict)} 组")
    except ImportError:
        # 降级为普通 PyTorch 保存
        final_path = final_path.replace(".safetensors", ".pt")
        lora_state_dict = {
            k: v for k, v in unet.state_dict().items()
            if "lora" in k.lower()
        }
        torch.save(lora_state_dict, final_path)
        print(f"\nLoRA 权重已保存: {final_path}")

    # 保存训练配置
    import json
    config_path = os.path.join(output_dir, f"{ethnicity}_training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "ethnicity": ethnicity,
            "name": config["name"],
            "trigger_word": config["trigger_word"],
            "base_model": BASE_MODEL_ID,
            "training_config": train_cfg,
            "final_loss": loss.item(),
            "total_steps": global_step,
        }, f, ensure_ascii=False, indent=2)

    print(f"训练配置已保存: {config_path}")
    return final_path


def train_all(
    data_dir: str = "dataset/lora_data",
    output_dir: str = "models/lora",
):
    """依次训练三个民族的 LoRA 模型"""
    results = {}
    for ethnicity in ["zang", "qiang", "yi"]:
        try:
            path = train_lora(ethnicity, data_dir, output_dir)
            results[ethnicity] = path
        except Exception as e:
            print(f"训练 {ethnicity} 失败: {e}")
            results[ethnicity] = None
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA 模型训练工具")
    parser.add_argument("--ethnicity", type=str, choices=["zang", "qiang", "yi", "all"],
                        default="all", help="要训练的民族")
    parser.add_argument("--data", type=str, default="dataset/lora_data", help="训练数据目录")
    parser.add_argument("--out", type=str, default="models/lora", help="输出目录")
    parser.add_argument("--steps", type=int, default=1500, help="训练步数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--resume", type=str, help="从检查点恢复")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_abs = project_root / args.data
    output_abs = project_root / args.out

    TRAINING_CONFIG["max_train_steps"] = args.steps
    TRAINING_CONFIG["learning_rate"] = args.lr

    if args.ethnicity == "all":
        train_all(str(data_abs), str(output_abs))
    else:
        train_lora(args.ethnicity, str(data_abs), str(output_abs),
                   resume_from_checkpoint=args.resume)
