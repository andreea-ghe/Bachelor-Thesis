"""
Batch-search for the best demo examples by randomly sampling objects from a
metadata pickle, running inference, and ranking them by assembly error.

Usage:
    python -m scripts.inference.find_best_examples \
        --cfg experiments/two_piece_scripts/everyday_eval.yaml \
        --metafile scripts/data/new-metadata-ds/two_fracture_assembly_metadata_2_2_everyday.val.txt \
        --sample_n 50 --seed 42

    python -m scripts.inference.find_best_examples \
        --cfg experiments/multi_piece_scripts/everyday_eval.yaml \
        --metafile scripts/data/new-metadata-ds/multi_fracture_assembly_metadata_2_4_everyday.val.txt \
        --sample_n 50 --seed 42
"""

import os
import csv
import pickle
import random
import argparse
import time

import numpy as np
import torch

from utilities.utils_config import CONFIG, config_from_file
from scripts.inference.predict_transformation import (
    load_model,
    load_meshes_from_dir,
    preprocess_meshes,
    compute_errors,
)


def load_metafile(metafile_path: str) -> list[str]:
    with open(metafile_path, "rb") as f:
        meta = pickle.load(f)
    return meta["data_list"]


def run_batch_search(args):
    config_from_file(args.cfg)
    torch.manual_seed(args.seed)

    model = load_model(CONFIG)

    all_paths = load_metafile(args.metafile)
    print(f"Metafile contains {len(all_paths)} entries")

    # filter to existing directories
    existing = [p for p in all_paths if os.path.isdir(p)]
    print(f"Found {len(existing)} existing directories (skipped {len(all_paths) - len(existing)})")

    if len(existing) == 0:
        print("No valid directories found. Are the paths in the metafile correct?")
        return

    # random sample
    sample_n = min(args.sample_n, len(existing))
    random.seed(args.seed)
    sampled = random.sample(existing, sample_n)
    print(f"Randomly sampled {sample_n} objects\n")

    results = []

    for idx, pieces_dir in enumerate(sampled):
        print(f"[{idx + 1}/{sample_n}] {pieces_dir}")

        try:
            meshes = load_meshes_from_dir(pieces_dir)
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"  SKIP: {e}\n")
            continue

        try:
            data_dict, transforms = preprocess_meshes(meshes, CONFIG, args.seed)

            with torch.no_grad():
                out_dict = model.forward(data_dict)
            pred_transforms = model.global_alignment(data_dict, out_dict)

            errors = compute_errors(pred_transforms, data_dict)
        except Exception as e:
            print(f"  FAIL: {e}\n")
            continue

        avg_rot = np.mean([e["rot_err_deg"] for e in errors])
        avg_trans = np.mean([e["trans_err"] for e in errors])
        max_rot = max(e["rot_err_deg"] for e in errors)
        max_trans = max(e["trans_err"] for e in errors)

        results.append({
            "path": pieces_dir,
            "num_pieces": len(meshes),
            "avg_rot_err_deg": round(avg_rot, 4),
            "avg_trans_err": round(avg_trans, 6),
            "max_rot_err_deg": round(max_rot, 4),
            "max_trans_err": round(max_trans, 6),
        })
        print()

    # sort by average rotation error (ascending = best first)
    results.sort(key=lambda r: r["avg_rot_err_deg"])

    # build output filename from metafile name
    meta_basename = os.path.splitext(os.path.basename(args.metafile))[0]
    cfg_basename = os.path.splitext(os.path.basename(args.cfg))[0]
    out_name = f"ranking_{cfg_basename}_{meta_basename}_n{sample_n}.csv"
    out_path = os.path.join("scripts", "inference", out_name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "rank", "path", "num_pieces",
        "avg_rot_err_deg", "avg_trans_err",
        "max_rot_err_deg", "max_trans_err",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(results, 1):
            writer.writerow({"rank": rank, **row})

    print("=" * 60)
    print(f"Ranked {len(results)} objects (best first), saved to: {out_path}")
    print("=" * 60)

    print(f"\nTop 10 best examples:")
    for i, r in enumerate(results[:10], 1):
        print(
            f"  {i:2d}. {r['path']}"
            f"\n      rot={r['avg_rot_err_deg']:7.2f}°  trans={r['avg_trans_err']:.4f}  "
            f"pieces={r['num_pieces']}"
        )

    if len(results) > 10:
        print(f"\nBottom 5 worst examples:")
        for r in results[-5:]:
            print(
                f"      {r['path']}"
                f"\n      rot={r['avg_rot_err_deg']:7.2f}°  trans={r['avg_trans_err']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best demo examples by batch inference on a metadata file")
    parser.add_argument("--cfg", required=True, help="Path to eval YAML config")
    parser.add_argument("--metafile", required=True, help="Metadata pickle (.txt) with data_list of folder paths")
    parser.add_argument("--sample_n", type=int, default=50, help="How many objects to randomly sample (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    run_batch_search(args)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
