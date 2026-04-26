#!/usr/bin/env python3

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
    folder = sys.argv[1]
    log_files = sorted(glob.glob(os.path.join(folder, "eval_log_*.log")))
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
