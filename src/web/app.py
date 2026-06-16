"""
Gradio 前端演示界面
双Tab界面：服饰识别 + AI绘图
"""
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from .prompts import get_prompt_guide


# 全局分类器实例
_classifier = None


def get_classifier():
    """延迟加载分类器"""
    global _classifier
    if _classifier is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / "models" / "classifier.pth"
        if model_path.exists():
            from src.classification.inference import CostumeClassifier
            _classifier = CostumeClassifier(str(model_path))
        else:
            return None
    return _classifier


# ============================================================
# 服饰识别 Tab
# ============================================================

def classify_fn(image: Optional[Image.Image]) -> dict:
    """分类回调函数"""
    if image is None:
        return {"藏族": 0.0, "羌族": 0.0, "彝族": 0.0}

    classifier = get_classifier()
    if classifier is None:
        return {"错误": 1.0, "藏族": 0.0, "羌族": 0.0, "彝族": 0.0}

    try:
        results = classifier.predict_from_pil(image, top_k=3)
        return {r["label"]: r["confidence"] for r in results}
    except Exception as e:
        return {"错误": 1.0, "藏族": 0.0, "羌族": 0.0, "彝族": 0.0}


# ============================================================
# AI 绘图 Tab
# ============================================================

def generate_fn(ethnicity: str, user_prompt: str, steps: int = 25,
                lora_scale: float = 0.8,
                custom_positive: str = "",
                custom_negative: str = "",
                sampler: str = "DPM++ 2M Karras") -> tuple[Optional[Image.Image], str, str]:
    """文生图回调函数 — 使用本地 SD + LoRA 推理"""
    if not ethnicity:
        return None, "请选择民族", ""

    from .inference import generate_local
    image, positive, negative = generate_local(
        ethnicity, user_prompt, lora_scale, steps,
        custom_positive=custom_positive,
        custom_negative=custom_negative,
        sampler=sampler)

    # 组合显示正向+负向提示词
    display = f"[正向提示词]\n{positive}\n\n[负向提示词]\n{negative}"
    return image, display, negative


# ============================================================
# 构建界面
# ============================================================

CSS = """
.container {max-width: 900px; margin: auto;}
.header {text-align: center; margin-bottom: 1rem;}
.footer {text-align: center; color: #888; margin-top: 2rem; font-size: 0.85rem;}
"""


def create_app(server_port: int = 8080, share: bool = False):
    """创建并启动 Gradio 应用"""

    # 加载提示词指南
    prompt_guide = get_prompt_guide()

    with gr.Blocks(title="民族服饰识别与AI绘图系统") as demo:
        gr.Markdown(
            """
            # 民族服饰识别与AI绘图系统
            ### 藏族 / 羌族 / 彝族 — 基于深度学习的服饰识别与风格化生成平台
            """,
            elem_classes=["header"],
        )

        with gr.Tabs():
            # ====================================================
            # Tab 1: 服饰识别
            # ====================================================
            with gr.Tab("服饰识别"):
                gr.Markdown(
                    """
                    **功能说明**：上传一张民族服饰图片，系统将自动识别其所属民族并给出置信度。
                    支持藏族、羌族、彝族三类传统服饰。
                    """
                )

                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(
                            type="pil", label="上传服饰图片",
                            sources=["upload", "clipboard"],
                        )
                        classify_btn = gr.Button("开始识别", variant="primary", size="lg")

                    with gr.Column():
                        result_output = gr.Label(
                            num_top_classes=3,
                            label="识别结果",
                        )

                classify_btn.click(
                    classify_fn,
                    inputs=[img_input],
                    outputs=[result_output],
                )

                gr.Markdown(
                    """
                    **提示**：
                    - 建议上传清晰的正面/侧面服饰图
                    - 支持 JPG/PNG/WEBP 格式
                    - 模型基于 ResNet-50 训练，验证准确率 ≥85%
                    """
                )

            # ====================================================
            # Tab 2: AI 绘图
            # ====================================================
            with gr.Tab("AI 绘图"):
                gr.Markdown(
                    """
                    **功能说明**：选择民族并输入画面描述，AI 将生成对应风格的民族服饰图片。
                    使用本地 Stable Diffusion + LoRA 推理，RTX 2060 约需 40-60 秒/张。
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        ethnicity_select = gr.Dropdown(
                            choices=["zang", "qiang", "yi"],
                            value="zang",
                            label="选择民族",
                            info="zang=藏族 | qiang=羌族 | yi=彝族",
                        )
                        steps_slider = gr.Slider(
                            10, 50, value=25, step=5,
                            label="采样步数",
                            info="步数越多质量越高，但生成更慢",
                        )
                        sampler_select = gr.Dropdown(
                            choices=[
                                "DPM++ 2M Karras",
                                "DPM++ SDE Karras",
                                "Euler a",
                                "Euler",
                                "DDIM",
                                "UniPC",
                                "PNDM",
                                "LMS",
                            ],
                            value="DPM++ 2M Karras",
                            label="采样器",
                            info="DPM++ 2M Karras 质量高速度快，Euler a 风格多样",
                        )
                        lora_slider = gr.Slider(
                            0.1, 1.5, value=0.8, step=0.1,
                            label="LoRA 权重",
                            info="越高越偏向目标风格，过高可能导致失真",
                        )
                        prompt_input = gr.Textbox(
                            label="画面描述（可选）",
                            placeholder="例如: 站在雪山前, 穿着节日盛装, 阳光下微笑...",
                            lines=2,
                        )
                        positive_input = gr.Textbox(
                            label="正向提示词（自定义，留空则使用预设）",
                            placeholder="留空使用预设。自定义示例: zang_costume, traditional Tibetan robe, masterpiece, 8k",
                            lines=3,
                        )
                        negative_input = gr.Textbox(
                            label="负向提示词（自定义，留空则使用默认）",
                            placeholder="留空使用默认。自定义示例: lowres, bad anatomy, blurry, modern clothes",
                            lines=3,
                        )
                        generate_btn = gr.Button("生成图片", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        img_output = gr.Image(
                            type="pil", label="生成结果",
                            format="png",
                        )
                        prompt_display = gr.Textbox(
                            label="实际使用的提示词",
                            lines=6,
                            interactive=False,
                        )

                generate_btn.click(
                    generate_fn,
                    inputs=[ethnicity_select, prompt_input, steps_slider, lora_slider,
                            positive_input, negative_input, sampler_select],
                    outputs=[img_output, prompt_display, gr.Textbox(visible=False)],
                )

                with gr.Accordion("提示词参考（点击展开）", open=False):
                    for eth, info in prompt_guide.items():
                        gr.Markdown(f"""
                        **{info['name']}** — 触发词: `{info['trigger_word']}`
                        - 风格关键词: {', '.join(f'`{kw}`' for kw in info['style_keywords'])}
                        - 示例: `{info['example_prompt']}`
                        ---
                        """)

            # ====================================================
            # Tab 3: 系统信息
            # ====================================================
            with gr.Tab("系统信息"):
                gr.Markdown(
                    """
                    ## 技术栈

                    | 组件 | 技术 |
                    |------|------|
                    | 分类模型 | ResNet-50 (ImageNet预训练 + 迁移学习) |
                    | 生成模型 | Stable Diffusion + LoRA |
                    | 后端框架 | FastAPI |
                    | 前端框架 | Gradio |
                    | 语言 | Python 3.10 |

                    ## 系统要求

                    - Python 3.10+
                    - CUDA GPU（推荐 4GB+ 显存）
                    - Stable Diffusion 1.5 + LoRA 本地推理

                    ## 项目结构

                    详见项目根目录的 README.md
                    """
                )

        gr.Markdown(
            "民族服饰识别与AI绘图系统 © 2024 — 数据科学与大数据技术综合实践项目",
            elem_classes=["footer"],
        )

    return demo


def launch_app(server_port: int = 8080, share: bool = False):
    """启动 Gradio 应用"""
    import os
    demo = create_app(server_port, share)
    print(f"\n启动 Web 演示界面...")
    print(f"本地访问: http://localhost:{server_port}")
    # Gradio 6.x 内部使用 requests 库验证启动，设置 no_proxy 防止代理干扰
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    demo.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        share=share,
        show_error=True,
        css=CSS,
        ssr_mode=False,
    )


if __name__ == "__main__":
    launch_app(8080, share=True)
