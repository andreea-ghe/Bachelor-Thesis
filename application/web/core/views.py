import json
import logging
from pathlib import Path

import httpx
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .fractures import get_fracture_by_id, get_fractures

logger = logging.getLogger(__name__)

MODEL_CHOICES = [
    ("two_piece_baseline", "Baseline (2 pieces)"),
    ("two_piece_baseline_incomplete", "Baseline (2 pieces, reduced)"),
    ("two_piece_pair_attn", "Pair attention (2 pieces)"),
    ("two_piece_pair_attn_incomplete", "Pair attention (2 pieces, reduced)"),
    ("two_piece_double_attn", "Double attention (2 pieces)"),
    ("two_piece_double_attn_incomplete", "Double attention (2 pieces, reduced)"),
    ("two_piece_gabriel_50", "Gabriel filtering (2 pieces, 50% retention)"),
    ("two_piece_gabriel_75", "Gabriel filtering (2 pieces, 75% retention)"),
    ("multi_piece_baseline", "Baseline (2-4 pieces)"),
    ("multi_piece_pair_attn", "Pair attention (2-4 pieces)"),
    ("multi_piece_double_attn", "Double attention (2-4 pieces)"),
]

PRECOMPUTED_PATH = (Path(__file__).resolve().parent / "static" / "core" / "precomputed_results.json")

_precomputed_cache: dict | None = None
def _get_precomputed() -> dict:
    global _precomputed_cache
    if _precomputed_cache is None:
        if PRECOMPUTED_PATH.exists():
            _precomputed_cache = json.loads(PRECOMPUTED_PATH.read_text())
        else:
            _precomputed_cache = {}
    return _precomputed_cache


def _lookup_precomputed(fracture_id: str, model: str) -> dict | None:
    cache = _get_precomputed()
    entry = cache.get(fracture_id)
    if entry and model in entry.get("models", {}):
        data = entry["models"][model]
        return {
            "model": model,
            "num_pieces": entry["num_pieces"],
            "pieces": data["pieces"],
            "combined_assembled_obj": data["combined_assembled_obj"],
            "source": "precomputed",
        }
    return None


@require_GET
def index(request):
    """Home page: list predefined fractures grouped by piece type."""
    fractures = get_fractures()
    two_piece = [f for f in fractures if f.piece_type == "two_pieces"]
    multi_piece = [f for f in fractures if f.piece_type == "multi_pieces"]

    return render(request, "core/index.html", {
        "two_piece_fractures": two_piece,
        "multi_piece_fractures": multi_piece,
        "model_choices": MODEL_CHOICES,
    })


@require_GET
def fracture_pieces(request, fracture_id):
    """Return the static URLs for a fracture's OBJ pieces."""
    fracture = get_fracture_by_id(fracture_id)
    if fracture is None:
        return JsonResponse({"error": "Fracture not found"}, status=404)

    pieces = []
    for obj_path in fracture.obj_paths:
        static_url = f"{settings.STATIC_URL}{fracture.static_prefix}/{obj_path.name}"
        pieces.append({"name": obj_path.stem, "url": static_url})

    return JsonResponse({
        "id": fracture.id,
        "display_name": fracture.display_name,
        "num_pieces": fracture.num_pieces,
        "piece_type": fracture.piece_type,
        "pieces": pieces,
    })


@require_POST
def predict(request):
    """
    Proxy inference to the GPU server; fall back to precomputed results
    if the server is unreachable or not configured.
    """
    fracture_id = request.POST.get("fracture_id")
    model_variant = request.POST.get("model", "two_piece_baseline")

    fracture = get_fracture_by_id(fracture_id)
    if fracture is None:
        return JsonResponse({"error": "Fracture not found"}, status=404)

    gpu_url = getattr(settings, "GPU_POD_URL", None)

    # Try live GPU first
    if gpu_url:
        files = []
        for obj_path in fracture.obj_paths:
            files.append(("pieces", (obj_path.name, obj_path.read_bytes(), "text/plain")))

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{gpu_url.rstrip('/')}/predict",
                    data={"model": model_variant},
                    files=files,
                )
            response.raise_for_status()
            result = response.json()
            result["source"] = "live"
            return JsonResponse(result)
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.warning("GPU unreachable, falling back to precomputed results")
        except httpx.HTTPStatusError as e:
            logger.warning("GPU error (%s), falling back to precomputed", e.response.status_code)

    # fallback to precomputed cache
    cached = _lookup_precomputed(fracture_id, model_variant)
    if cached is not None:
        return JsonResponse(cached)

    is_two_piece_model = model_variant.startswith("two_piece")
    if is_two_piece_model and fracture.num_pieces > 2:
        return JsonResponse({
            "error": (
                f"The selected model supports only 2 pieces, but this object "
                f"has {fracture.num_pieces} fragments. Please select a multi-piece "
                f"model (2-4 pieces) from the dropdown."
            )
        }, status=400)

    return JsonResponse({"error": "GPU is unavailable and no precomputed result exists for this combination."}, status=503)
