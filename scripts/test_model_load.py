"""验证 SD 1.5 模型加载"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_PATH = str(PROJECT_ROOT / "sd")

print("测试加载 tokenizer...")
from transformers import CLIPTokenizer
tok = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
print(f"  tokenizer OK, vocab_size={tok.vocab_size}")

print("测试加载 text_encoder...")
from transformers import CLIPTextModel
te = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder")
print(f"  text_encoder OK, params={sum(p.numel() for p in te.parameters()):,}")

print("测试加载 unet...")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet")
print(f"  unet OK, params={sum(p.numel() for p in unet.parameters()):,}")

print("测试加载 scheduler...")
from diffusers import DDPMScheduler
sched = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
print(f"  scheduler OK, timesteps={sched.config.num_train_timesteps}")

print("\n全部模型组件加载成功!")
