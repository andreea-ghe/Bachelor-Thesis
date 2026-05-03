import logging

import httpx
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .fractures import get_fracture_by_id, get_fractures

logger = logging.getLogger(__name__)

MODEL_CHOICES = [
    ("two_piece_baseline", "Baseline (2 pieces)"),
    ("two_piece_pair_attn", "Pair attention (2 pieces)"),
    ("two_piece_double_attn", "Double attention (2 pieces)"),
    ("multi_piece_baseline", "Baseline (2-4 pieces)"),
    ("multi_piece_pair_attn", "Pair attention (2-4 pieces)"),
    ("multi_piece_double_attn", "Double attention (2-4 pieces)"),
]


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
    Proxy inference request to the FastAPI GPU server.

    Reads the OBJ files for the selected fracture, sends them as multipart
    to the inference server, and returns the assembled result.
    """
    gpu_url = settings.GPU_POD_URL
    if not gpu_url:
        return JsonResponse({"error": "GPU_POD_URL is not configured"}, status=503)

    fracture_id = request.POST.get("fracture_id")
    model_variant = request.POST.get("model", "two_piece_baseline")

    fracture = get_fracture_by_id(fracture_id)
    if fracture is None:
        return JsonResponse({"error": "Fracture not found"}, status=404)

    # Read OBJ files from disk
    files = []
    for obj_path in fracture.obj_paths:
        files.append(("pieces", (obj_path.name, obj_path.read_bytes(), "text/plain")))

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{gpu_url}/predict",
                data={"model": model_variant},
                files=files,
            )
        response.raise_for_status()
        result = response.json()
    except httpx.ConnectError:
        logger.error("Cannot reach inference server at %s", gpu_url)
        return JsonResponse({"error": "Inference server is not reachable"}, status=503)
    except httpx.TimeoutException:
        return JsonResponse({"error": "Inference timed out"}, status=504)
    except httpx.HTTPStatusError as e:
        logger.error("Inference server error: %s", e.response.text)
        return JsonResponse({"error": f"Inference failed: {e.response.text}"}, status=502)

    return JsonResponse(result)
