"""
Compare two dataset metadata files.
"""
import pickle
import argparse
from collections import Counter


def load_metadata(path):
    with open(path, 'rb') as f:
        meta = pickle.load(f)
    return meta['data_list']


def extract_mesh(entry):
    """Extract mesh identifier (Category/hash) from a data_list entry like 'everyday/Vase/hash/fractured_3'."""
    parts = entry.split('/')
    for i, p in enumerate(parts):
        if p and p[0].isupper() and i + 1 < len(parts):
            return '/'.join(parts[i:i+2])
    return entry


def extract_category(entry):
    parts = entry.split('/')
    for p in parts:
        if p and p[0].isupper():
            return p
    return 'Unknown'


def compare(old_path, new_path):
    old_list = load_metadata(old_path)
    new_list = load_metadata(new_path)

    old_set = set(old_list)
    new_set = set(new_list)

    print(f"Old dataset: {len(old_list)} fracture patterns")
    print(f"New dataset: {len(new_list)} fracture patterns")
    print(f"Difference:  {len(new_list) - len(old_list)} fracture patterns")
    print()

    missing_patterns = new_set - old_set
    extra_patterns = old_set - new_set

    old_meshes = set(extract_mesh(e) for e in old_list)
    new_meshes = set(extract_mesh(e) for e in new_list)
    missing_meshes = new_meshes - old_meshes

    print(f"Meshes in old: {len(old_meshes)}")
    print(f"Meshes in new: {len(new_meshes)}")
    print(f"Meshes missing from old: {len(missing_meshes)}")
    print()

    if missing_patterns:
        print(f"--- Fracture patterns in new but NOT in old: {len(missing_patterns)} ---")
        cats = Counter(extract_category(e) for e in missing_patterns)
        total_cats = Counter(extract_category(e) for e in new_list)
        for cat, count in cats.most_common():
            print(f"  {cat}: {count}/{total_cats[cat]} patterns missing ({100*count/total_cats[cat]:.1f}%)")
        print()

    if missing_meshes:
        print(f"--- Meshes in new but NOT in old: {len(missing_meshes)} ---")
        cats = Counter(m.split('/')[0] for m in missing_meshes)
        total_cats = Counter(extract_mesh(e).split('/')[0] for e in new_list)
        for cat, count in cats.most_common():
            print(f"  {cat}: {count}/{total_cats[cat]} meshes missing ({100*count/total_cats[cat]:.1f}%)")
        print()
        print("Missing mesh IDs:")
        for mesh in sorted(missing_meshes):
            print(f"  {mesh}")

    if extra_patterns:
        print(f"\n--- Fracture patterns in old but NOT in new: {len(extra_patterns)} ---")
        cats = Counter(extract_category(e) for e in extra_patterns)
        for cat, count in cats.most_common():
            print(f"  {cat}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two dataset metadata files')
    parser.add_argument('--old', required=True, help='Path to old (incomplete) metadata pickle')
    parser.add_argument('--new', required=True, help='Path to new (complete) metadata pickle')
    args = parser.parse_args()

    compare(args.old, args.new)
