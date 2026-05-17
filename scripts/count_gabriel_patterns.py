"""Count 2-piece fracture patterns available for the reduced dataset splits."""
import os
import sys
from collections import defaultdict


def load_mesh_list(path):
    """Load mesh directory names from a metadata file."""
    with open(path, 'r') as f:
        return set(line.strip().rstrip('/').split('/')[-1] for line in f if line.strip())


def count_patterns_from_csv(csv_path, mesh_set):
    """Count unique fracture patterns in the CSV that belong to the given mesh set."""
    matched = set()
    total = 0
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            # Format: Category_meshhash_fractured_N_piece_0.obj,Category_meshhash_fractured_N_piece_1.obj
            piece0 = line.split(',')[0]
            # Extract: Category_meshhash from Category_meshhash_fractured_N_piece_0.obj
            parts = piece0.replace('.obj', '').split('_')
            # Find the 'fractured' keyword to split mesh name from fracture info
            frac_idx = parts.index('fractured')
            mesh_name = '_'.join(parts[:frac_idx])
            fracture_id = '_'.join(parts[:frac_idx + 2])  # Category_hash_fractured_N

            if mesh_name in mesh_set:
                matched.add(fracture_id)

    return matched, total


def count_patterns_from_disk(meta_path, data_dir):
    """Count 2-piece fracture patterns by scanning disk."""
    count = 0
    with open(meta_path, 'r') as f:
        meshes = [line.strip() for line in f if line.strip()]

    for mesh in meshes:
        mesh_dir = os.path.join(data_dir, mesh)
        if not os.path.isdir(mesh_dir):
            continue
        for entry in os.listdir(mesh_dir):
            frac_dir = os.path.join(mesh_dir, entry)
            if entry.startswith('fractured_') and os.path.isdir(frac_dir):
                pieces = [p for p in os.listdir(frac_dir) if p.endswith('.obj')]
                if len(pieces) == 2:
                    count += 1
    return count


if __name__ == '__main__':
    data_dir = '/workspace'

    train_meta = os.path.join(data_dir, 'original_everyday.train.txt')
    val_meta = os.path.join(data_dir, 'original_everyday.val.txt')
    csv_path = os.path.join(data_dir, 'everyday_all_2', 'everyday_pairs_all_2.csv')

    print("=" * 60)
    print("Reduced dataset: mesh counts")
    print("=" * 60)

    train_meshes = load_mesh_list(train_meta)
    val_meshes = load_mesh_list(val_meta)
    print(f"  Train meshes: {len(train_meshes)}")
    print(f"  Val meshes:   {len(val_meshes)}")

    print()
    print("=" * 60)
    print("Method 1: Cross-reference CSV with reduced mesh lists")
    print("=" * 60)

    train_patterns, csv_total = count_patterns_from_csv(csv_path, train_meshes)
    val_patterns, _ = count_patterns_from_csv(csv_path, val_meshes)
    print(f"  CSV total lines:          {csv_total}")
    print(f"  2-piece train patterns:   {len(train_patterns)}")
    print(f"  2-piece val patterns:     {len(val_patterns)}")
    print(f"  Combined:                 {len(train_patterns) + len(val_patterns)}")

    print()
    print("=" * 60)
    print("Method 2: Scan disk for fractured_* dirs with exactly 2 .obj files")
    print("=" * 60)

    train_disk = count_patterns_from_disk(train_meta, data_dir)
    val_disk = count_patterns_from_disk(val_meta, data_dir)
    print(f"  2-piece train patterns:   {train_disk}")
    print(f"  2-piece val patterns:     {val_disk}")
    print(f"  Combined:                 {train_disk + val_disk}")

    print()
    print("These are the samples your Gabriel experiment will train on")
    print(f"(with MAX_NUM_PART=2 and BATCH_SIZE=4 → ~{train_disk // 4} batches/epoch)")
