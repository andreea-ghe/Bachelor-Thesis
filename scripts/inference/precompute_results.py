"""
Pre-compute inference results for every fracture x model combination.

Usage:
    python -m scripts.inference.precompute_results --gpu_url http://<pod-ip>:8000 [--output application/web/core/static/core/precomputed_results.json] [--timeout 180]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from application.web.core.fractures import discover_fractures

TWO_PIECE_MODELS = [
    "two_piece_baseline",
    "two_piece_baseline_incomplete",
    "two_piece_pair_attn",
    "two_piece_pair_attn_incomplete",
    "two_piece_double_attn",
    "two_piece_double_attn_incomplete",
    "two_piece_gabriel_50",
    "two_piece_gabriel_75",
]

MULTI_PIECE_MODELS = [
    "multi_piece_baseline",
    "multi_piece_pair_attn",
    "multi_piece_double_attn",
]

ALL_MODELS = TWO_PIECE_MODELS + MULTI_PIECE_MODELS


def get_compatible_models(num_pieces: int) -> list[str]:
    if num_pieces == 2:
        return ALL_MODELS
    return MULTI_PIECE_MODELS


def run_single(client: httpx.Client, gpu_url: str, fracture, model: str) -> dict | None:
    files = []
    for obj_path in fracture.obj_paths:
        files.append(("pieces", (obj_path.name, obj_path.read_bytes(), "text/plain")))

    try:
        resp = client.post(
            f"{gpu_url.rstrip('/')}/predict",
            data={"model": model},
            files=files,
        )
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
        print(f"    FAILED: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pre-compute all fracture × model results")
    parser.add_argument("--gpu_url", required=True, help="Base URL of the inference server")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "application" / "web" / "core" / "static" / "core" / "precomputed_results.json"),
        help="Output JSON path",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Skip combinations already in existing output file")
    args = parser.parse_args()

    fractures = discover_fractures()
    print(f"Found {len(fractures)} fractures")

    total_combos = sum(len(get_compatible_models(f.num_pieces)) for f in fractures)
    print(f"Total combinations to compute: {total_combos}")

    existing = {}
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} fracture entries already cached")

    results = dict(existing)
    done = 0
    failed = 0

    with httpx.Client(timeout=args.timeout) as client:
        for i, fracture in enumerate(fractures):
            models = get_compatible_models(fracture.num_pieces)
            frac_key = fracture.id

            if frac_key not in results:
                results[frac_key] = {
                    "id": fracture.id,
                    "display_name": fracture.display_name,
                    "category": fracture.category,
                    "num_pieces": fracture.num_pieces,
                    "piece_type": fracture.piece_type,
                    "models": {},
                }

            for model in models:
                combo_label = f"[{done + failed + 1}/{total_combos}] {fracture.display_name} x {model}"

                if model in results[frac_key].get("models", {}):
                    print(f"  SKIP (cached): {combo_label}")
                    done += 1
                    continue

                print(f"  {combo_label} ...", end=" ", flush=True)
                t0 = time.time()
                result = run_single(client, args.gpu_url, fracture, model)
                elapsed = time.time() - t0

                if result is not None:
                    results[frac_key]["models"][model] = {
                        "pieces": result["pieces"],
                        "combined_assembled_obj": result["combined_assembled_obj"],
                    }
                    done += 1
                    print(f"OK ({elapsed:.1f}s)")
                else:
                    failed += 1

            # save after each fracture so progress isn't lost
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f)

    print(f"\nDone: {done} succeeded, {failed} failed")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
