"""
FastAPI 后端服务
提供 /classify（图像识别）和 /generate（文生图）两个核心接口
"""
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from .prompts import get_prompt_guide

app = FastAPI(
    title="民族服饰识别与AI绘图系统",
    description="基于深度学习的藏族/羌族/彝族服饰识别与风格化生成平台",
    version="1.0.0",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 延迟加载分类器
_classifier = None


def get_classifier():
    """获取全局分类器实例"""
    global _classifier
    if _classifier is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / "models" / "classifier.pth"

        if model_path.exists():
            from src.classification.inference import CostumeClassifier
            _classifier = CostumeClassifier(str(model_path))
        else:
            raise HTTPException(status_code=503, detail="分类模型尚未训练，请先运行训练脚本")
    return _classifier


# ============================================================
# 请求/响应模型
# ============================================================

class ClassifyResponse(BaseModel):
    label: str = Field(..., description="预测的民族类别")
    confidence: float = Field(..., description="置信度")
    top3: list[dict] = Field(default_factory=list, description="前三预测结果")


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", description="自定义描述文本")
    ethnicity: str = Field(default="zang", description="民族: zang/qiang/yi")
    steps: int = Field(default=25, ge=10, le=50, description="采样步数")
    lora_scale: float = Field(default=0.8, ge=0.1, le=1.5, description="LoRA权重")
    cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0, description="CFG强度")
    custom_positive: str = Field(default="", description="自定义正向提示词（留空使用预设）")
    custom_negative: str = Field(default="", description="自定义负向提示词（留空使用默认）")
    sampler: str = Field(default="DPM++ 2M Karras", description="采样器名称")


class GenerateResponse(BaseModel):
    image_base64: str = Field(..., description="生成的图片(base64编码)")
    positive_prompt: str = Field(..., description="实际使用的正向提示词")
    negative_prompt: str = Field(..., description="使用的负向提示词")
    parameters: dict = Field(default_factory=dict, description="生成参数")


class HealthResponse(BaseModel):
    status: str
    classifier_available: bool
    lora_models: list[str]


# ============================================================
# 接口
# ============================================================

@app.get("/", response_model=dict)
async def root():
    """API 根路由"""
    return {
        "message": "民族服饰识别与AI绘图系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/classify", "/generate", "/health", "/prompts"],
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """系统健康检查"""
    project_root = Path(__file__).resolve().parent.parent.parent

    # 检查分类模型
    classifier_path = project_root / "models" / "classifier.pth"
    classifier_available = classifier_path.exists()

    # 检查 LoRA 模型
    lora_dir = project_root / "models" / "lora"
    lora_models = []
    if lora_dir.exists():
        for f in lora_dir.glob("*_lora.*"):
            lora_models.append(f.name)

    return HealthResponse(
        status="ok" if classifier_available or lora_models else "degraded",
        classifier_available=classifier_available,
        lora_models=lora_models,
    )


@app.get("/prompts")
async def get_prompts():
    """获取提示词指南"""
    return {
        "guide": get_prompt_guide(),
        "negative_prompt": "详见源码中的 NEGATIVE_PROMPT 常量",
        "default_parameters": {
            "steps": 25,
            "cfg_scale": 7.0,
            "lora_scale": 0.8,
            "sampler": "DPM++ 2M Karras",
        },
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(..., description="待识别的服饰图片")):
    """
    图像识别接口：上传服饰图片，返回民族类别与置信度
    """
    # 验证文件类型
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(status_code=400, detail="请上传 JPEG/PNG/WEBP/BMP 格式的图片")

    # 读取图片
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析图片文件，请确认文件完整")

    # 推理
    classifier = get_classifier()
    results = classifier.predict_from_pil(img, top_k=3)

    return ClassifyResponse(
        label=results[0]["label"],
        confidence=results[0]["confidence"],
        top3=results,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    文生图接口：根据民族选择和文本描述生成服饰图片
    使用本地 Stable Diffusion + LoRA 推理
    """
    if req.ethnicity not in ("zang", "qiang", "yi"):
        raise HTTPException(status_code=400, detail="民族参数必须是: zang, qiang, yi")

    from .inference import generate_local

    img, positive, negative = generate_local(
        req.ethnicity, req.prompt,
        lora_scale=req.lora_scale, steps=req.steps, cfg_scale=req.cfg_scale,
        custom_positive=req.custom_positive,
        custom_negative=req.custom_negative,
        sampler=req.sampler,
    )

    if img is None:
        raise HTTPException(status_code=503, detail=positive)

    # 图片转 base64
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return GenerateResponse(
        image_base64=image_b64,
        positive_prompt=positive,
        negative_prompt=negative,
        parameters={
            "ethnicity": req.ethnicity,
            "steps": req.steps,
            "cfg_scale": req.cfg_scale,
            "lora_scale": req.lora_scale,
            "sampler": req.sampler,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
