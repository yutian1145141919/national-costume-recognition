"""从 ModelScope 下载 Stable Diffusion 1.5 模型"""
from modelscope import snapshot_download

model_dir = snapshot_download(
    'AI-ModelScope/stable-diffusion-v1-5',
    cache_dir='D:/pbl/民族lora/models/base_models'
)
print(f'模型已下载到: {model_dir}')
