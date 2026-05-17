"""Count 2-piece fracture patterns and inspect pre-computed caches."""
import os
import pickle


def try_load_pickle(path):
    """Try loading a file as a pickle cache (data loader format)."""
    try:
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        if isinstance(meta, dict) and 'data_list' in meta:
            data_list = meta['data_list']
            print(f"  → Pickle cache with {len(data_list)} patterns")
            if data_list:
                print(f"  → Sample entry: {data_list[0]}")
            return len(data_list)
        else:
            print(f"  → Pickle loaded but unexpected format: {type(meta)}")
            return None
    except Exception:
        # Not a pickle, try as text
        try:
            with open(path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            print(f"  → Text file with {len(lines)} lines")
            if lines:
                print(f"  → Sample line: {lines[0]}")
            return len(lines)
        except Exception as e:
            print(f"  → Cannot read: {e}")
            return None


def count_patterns_from_disk(meta_path, data_dir):
    """Count 2-piece fracture patterns by scanning disk."""
    count = 0
    with open(meta_path, 'r') as f:
        meshes = [line.strip() for line in f if line.strip()]
    for mesh in meshes:
        mesh_dir = os.path.join(data_dir, mesh)
        if not os.path.isdir(mesh_dir):
            continue
        for entry in sorted(os.listdir(mesh_dir)):
            frac_dir = os.path.join(mesh_dir, entry)
            if entry.startswith('fractured_') and os.path.isdir(frac_dir):
                pieces = [p for p in os.listdir(frac_dir) if p.endswith('.obj')]
                if len(pieces) == 2:
                    count += 1
    return count


if __name__ == '__main__':
    data_dir = '/workspace'

    print("=" * 60)
    print("1. Check ALL pre-computed cache files at /workspace/")
    print("=" * 60)

    cache_candidates = [
        'fracture_assembly_metadata_2_2_everyday.train.txt',
        'fracture_assembly_metadata_2_2_everyday.val.txt',
        'original_fracture_assembly_metadata_2_2_everyday.train.txt',
        'original_fracture_assembly_metadata_2_2_everyday.val.txt',
        'fracture_assembly_metadata_2_2_artifact.train.txt',
        'fracture_assembly_metadata_2_2_artifact.val.txt',
        'metadata_train_this_server.pkl',
    ]

    for name in cache_candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"\n  [{name}] ({size:,} bytes)")
            try_load_pickle(path)
        else:
            print(f"\n  [{name}] NOT FOUND")

    # Also check everyday_all_2 directory
    all2_dir = os.path.join(data_dir, 'everyday_all_2')
    if os.path.isdir(all2_dir):
        print(f"\n  --- Files in everyday_all_2/ ---")
        for name in sorted(os.listdir(all2_dir)):
            path = os.path.join(all2_dir, name)
            size = os.path.getsize(path)
            print(f"\n  [everyday_all_2/{name}] ({size:,} bytes)")
            try_load_pickle(path)

    print()
    print("=" * 60)
    print("2. Disk scan: 2-piece patterns per reduced dataset split")
    print("=" * 60)

    train_meta = os.path.join(data_dir, 'original_everyday.train.txt')
    val_meta = os.path.join(data_dir, 'original_everyday.val.txt')

    if os.path.exists(train_meta):
        train_disk = count_patterns_from_disk(train_meta, data_dir)
        print(f"  Reduced train (from original_everyday.train.txt): {train_disk}")
    if os.path.exists(val_meta):
        val_disk = count_patterns_from_disk(val_meta, data_dir)
        print(f"  Reduced val (from original_everyday.val.txt):     {val_disk}")
        print(f"  Combined: {train_disk + val_disk}")

    print()
    print("=" * 60)
    print("3. Summary")
    print("=" * 60)
    print("  Compare the pickle cache counts with the disk scan counts.")
    print("  If a pickle cache exists for your DATA_FN, the data loader")
    print("  uses it DIRECTLY instead of scanning disk.")
    print("  Your previous experiments used whichever cache was present.")
