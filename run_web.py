#!/usr/bin/env python
"""
Web 演示系统 — 独立入口
启动 Gradio 前端界面（包含 FastAPI 后端路由）

使用方法:
    conda activate costume310
    python run_web.py --port 8080
"""
import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# 环境检测：确保使用正确的 Python 环境
# ============================================================
_CONDA_ENV_NAME = "costume310"
_CONDA_ENV_PATHS = [
    Path(os.environ.get("CONDA_EXE", "")).parent.parent / "envs" / _CONDA_ENV_NAME,
    Path("D:/anaconda3/envs") / _CONDA_ENV_NAME,
    Path.home() / "miniconda3" / "envs" / _CONDA_ENV_NAME,
    Path.home() / "anaconda3" / "envs" / _CONDA_ENV_NAME,
]

_current_py = Path(sys.executable).resolve()


def _is_valid_env():
    """检查当前 Python 是否符合项目要求"""
    py_ver = sys.version_info[:2]
    if py_ver != (3, 10):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    return True


def _find_conda_python():
    """查找 costume310 conda 环境的 Python 路径"""
    for base in _CONDA_ENV_PATHS:
        py = base / "python.exe"
        if py.exists():
            return py
    return None


def _relaunch_in_env():
    """在当前 conda 环境中重新启动自身"""
    conda_py = _find_conda_python()
    if conda_py is None:
        print(f"[错误] 未找到 conda 环境 '{_CONDA_ENV_NAME}'")
        print(f"  请先运行: conda create -n {_CONDA_ENV_NAME} python=3.10 -y")
        print(f"  并按 docs/deployment.md 配置环境")
        sys.exit(1)

    print(f"检测到当前 Python 不匹配，正在切换到: {conda_py}")
    # 用正确的 Python 重新执行当前脚本，透传参数
    args = [str(conda_py), __file__] + sys.argv[1:]
    result = subprocess.run(args)
    sys.exit(result.returncode)


if not _is_valid_env():
    _relaunch_in_env()

sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="民族服饰AI系统 Web 演示")
    parser.add_argument("--port", type=int, default=8080, help="服务端口（默认8080）")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    parser.add_argument("--api-only", action="store_true", help="只启动FastAPI后端（不含Gradio）")
    args = parser.parse_args()

    print("=" * 50)
    print("民族服饰识别与AI绘图 Web 演示系统")
    print("=" * 50)

    # 显示运行环境
    import torch
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 检查模型
    classifier_path = PROJECT_ROOT / "models" / "classifier.pth"
    print(f"\n分类模型: {'已就绪' if classifier_path.exists() else '未找到（识别功能不可用）'}")
    print(f"  {classifier_path}")

    lora_dir = PROJECT_ROOT / "models" / "lora"
    lora_files = list(lora_dir.glob("*_lora.*"))
    if lora_files:
        print(f"LoRA 模型: {len(lora_files)} 个")
        for f in lora_files:
            print(f"  {f.name}")
    else:
        print("LoRA 模型: 未找到（文生图功能需要单独配置）")

    if args.api_only:
        import uvicorn
        from src.web.api import app
        print(f"\nFastAPI 服务启动: http://localhost:{args.port}")
        print(f"API 文档: http://localhost:{args.port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        from src.web.app import launch_app
        launch_app(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
