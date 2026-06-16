"""测试 EasyNegative 嵌入加载"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from web.inference import _load_pipeline

print("加载模型和嵌入...")
pipe = _load_pipeline()
print("完成!")
