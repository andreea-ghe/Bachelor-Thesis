import os
import sys
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class ModelVariant(str, Enum):
    two_piece_baseline = "two_piece_baseline"
    two_piece_pair_attn = "two_piece_pair_attn"
    two_piece_double_attn = "two_piece_double_attn"
    multi_piece_baseline = "multi_piece_baseline"
    multi_piece_pair_attn = "multi_piece_pair_attn"
    multi_piece_double_attn = "multi_piece_double_attn"


MODEL_CONFIGS = {
    ModelVariant.two_piece_baseline: "experiments/two_piece_scripts/everyday_eval.yaml",
    ModelVariant.two_piece_pair_attn: "experiments/pair_attn_scripts/everyday_eval.yaml",
    ModelVariant.two_piece_double_attn: "experiments/double_attn_scripts/everyday_eval.yaml",
    ModelVariant.multi_piece_baseline: "experiments/multi_piece_scripts/everyday_eval.yaml",
    ModelVariant.multi_piece_pair_attn: "experiments/multi_pair_attn_scripts/everyday_eval.yaml",
    ModelVariant.multi_piece_double_attn: "experiments/multi_double_attn_scripts/everyday_eval.yaml",
}

MODEL_DESCRIPTIONS = {
    ModelVariant.two_piece_baseline: "Jigsaw baseline (2 pieces)",
    ModelVariant.two_piece_pair_attn: "Pair geometric attention (2 pieces)",
    ModelVariant.two_piece_double_attn: "Double iterative attention (2 pieces)",
    ModelVariant.multi_piece_baseline: "Jigsaw baseline (2-4 pieces)",
    ModelVariant.multi_piece_pair_attn: "Pair geometric attention (2-4 pieces)",
    ModelVariant.multi_piece_double_attn: "Double iterative attention (2-4 pieces)",
}

MODEL_MAX_PIECES = {
    ModelVariant.two_piece_baseline: 2,
    ModelVariant.two_piece_pair_attn: 2,
    ModelVariant.two_piece_double_attn: 2,
    ModelVariant.multi_piece_baseline: 4,
    ModelVariant.multi_piece_pair_attn: 4,
    ModelVariant.multi_piece_double_attn: 4,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load a default model at startup."""
    preload = os.environ.get("PRELOAD_MODEL")
    if preload and preload in ModelVariant.__members__:
        from application.inference_server.run_inference import _get_model

        cfg = os.path.join(REPO_ROOT, MODEL_CONFIGS[ModelVariant(preload)])
        _get_model(cfg)
    yield


app = FastAPI(
    title="Jigsaw Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("ALLOWED_ORIGIN", "*")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    import torch

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/models")
async def list_models():
    """Return available model variants and their config flags."""
    return {
        "models": [
            {
                "id": v.value,
                "description": MODEL_DESCRIPTIONS[v],
                "max_pieces": MODEL_MAX_PIECES[v],
            }
            for v in ModelVariant
        ]
    }


@app.post("/predict")
async def predict(
    model: ModelVariant = Form(ModelVariant.two_piece_baseline),
    pieces: list[UploadFile] = File(..., description="OBJ files for each piece"),
):
    """
    Run inference on uploaded OBJ pieces.

    - **model**: which model variant to use
    - **pieces**: 2-4 OBJ files (multipart upload)

    Returns per-piece assembled OBJs and a combined mesh (all as OBJ text).
    """
    max_pieces = MODEL_MAX_PIECES[model]
    if len(pieces) < 2:
        raise HTTPException(400, "At least 2 pieces are required.")
    if len(pieces) > max_pieces:
        raise HTTPException(400, f"Model '{model.value}' supports at most {max_pieces} pieces, got {len(pieces)}.")

    obj_strings = []
    for piece in pieces:
        raw = await piece.read()
        try:
            obj_strings.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise HTTPException(400, f"File '{piece.filename}' is not valid UTF-8 text.")

    config_path = os.path.join(REPO_ROOT, MODEL_CONFIGS[model])
    if not os.path.exists(config_path):
        raise HTTPException(500, f"Config not found: {MODEL_CONFIGS[model]}")

    try:
        from application.inference_server.run_inference import run_inference

        result = run_inference(config_path, obj_strings)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    return JSONResponse({
        "model": model.value,
        "num_pieces": len(pieces),
        "pieces": [
            {"name": p["name"], "assembled_obj": p["assembled_obj"]}
            for p in result["pieces"]
        ],
        "combined_assembled_obj": result["combined_assembled_obj"],
    })
