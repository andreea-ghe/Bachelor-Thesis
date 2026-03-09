#!/usr/bin/env python3
"""
Extract test metrics from eval log files and format them as a Python list
ready to paste into avg_results.py.

Usage:
    python scripts/extract_eval_results.py <folder_path>

Example:
    python scripts/extract_eval_results.py results/jigsaw_multi_4x4_128_512_250e_cosine_everyday/double_layers_new_arch/simple_attn
"""
import sys
import os
import glob
import re


def extract_metrics_from_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()

    pattern = r'test/\w+: [\d.e+-]+'
    matches = re.findall(pattern, content)
    if not matches:
        return None

    seen = {}
    for match in matches:
        key, value = match.split(': ', 1)
        seen[key] = float(value)

    return seen


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_eval_results.py <folder_path>")
        print("\nAvailable folders:")
        base_dirs = [
            "results/jigsaw_multi_4x4_128_512_250e_cosine_everyday/double_layers_new_arch/simple_attn",
            "results/jigsaw_multi_4x4_128_512_250e_cosine_everyday/double_layers_new_arch/pair_attn",
            "results/jigsaw_multi_4x4_128_512_250e_cosine_artifact/double_layers_new_arch/simple_attn",
            "results/jigsaw_multi_4x4_128_512_250e_cosine_artifact/double_layers_new_arch/pair_attn",
        ]
        for d in base_dirs:
            if os.path.isdir(d):
                n = len(glob.glob(os.path.join(d, "eval_log_*.log")))
                print(f"  {d}  ({n} logs)")
        sys.exit(1)

    folder = sys.argv[1]
    log_files = sorted(glob.glob(os.path.join(folder, "eval_log_*.log")))

    if not log_files:
        print(f"No eval_log_*.log files found in {folder}")
        sys.exit(1)

    print(f"Found {len(log_files)} eval logs in {folder}\n")

    results = []
    for log_path in log_files:
        metrics = extract_metrics_from_log(log_path)
        if metrics:
            results.append(metrics)
        else:
            print(f"WARNING: No metrics found in {os.path.basename(log_path)}")

    print(f"list_of_results = [")
    for i, r in enumerate(results):
        print("  {")
        for j, (key, value) in enumerate(sorted(r.items())):
            comma = "," if j < len(r) - 1 else ""
            print(f'      "{key}": {value}{comma}')
        comma = "," if i < len(results) - 1 else ""
        print(f"  }}{comma}")
    print("]")


if __name__ == "__main__":
    main()
