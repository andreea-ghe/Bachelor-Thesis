import hashlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MESHES_DIR = Path(__file__).resolve().parent.parent.parent / "application" / "web" / "core" / "static" / "core" / "meshes"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "application" / "web" / "core" / "static" / "core" / "thumbnails"

PIECE_COLORS = ["#00c850", "#c80000", "#0000c8", "#c8c800"]


def make_id(piece_type: str, category: str, mesh_hash: str, fracture: str) -> str:
    raw = f"{piece_type}/{category}/{mesh_hash}/{fracture}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_obj_vertices_faces(path: Path):
    vertices = []
    faces = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            face_verts = []
            for p in parts[1:]:
                idx = int(p.split("/")[0]) - 1
                face_verts.append(idx)
            if len(face_verts) >= 3:
                faces.append(face_verts)
    return np.array(vertices) if vertices else np.zeros((0, 3)), faces


def subsample_faces(faces, max_faces=2000):
    if len(faces) <= max_faces:
        return faces
    indices = np.random.choice(len(faces), max_faces, replace=False)
    return [faces[i] for i in indices]


def render_thumbnail(obj_paths: list[Path], output_path: Path):
    fig = plt.figure(figsize=(2.5, 2), dpi=96)
    ax = fig.add_subplot(111, projection="3d")

    all_verts = []
    for i, obj_path in enumerate(obj_paths):
        verts, faces = load_obj_vertices_faces(obj_path)
        if len(verts) == 0:
            continue
        all_verts.append(verts)
        color = PIECE_COLORS[i % len(PIECE_COLORS)]
        faces = subsample_faces(faces)
        polygons = [[verts[vi] for vi in f] for f in faces]
        collection = Poly3DCollection(polygons, alpha=0.85, linewidths=0.1, edgecolors="#222222")
        collection.set_facecolor(color)
        ax.add_collection3d(collection)

    if not all_verts:
        plt.close(fig)
        return

    combined = np.vstack(all_verts)
    center = combined.mean(axis=0)
    extent = max(combined.max(axis=0) - combined.min(axis=0)) * 0.6

    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=135)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0

    for piece_type in ("two_pieces", "multi_pieces"):
        type_dir = MESHES_DIR / piece_type
        if not type_dir.is_dir():
            continue

        for category_dir in sorted(type_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            for hash_dir in sorted(category_dir.iterdir()):
                if not hash_dir.is_dir():
                    continue

                for fracture_dir in sorted(hash_dir.iterdir()):
                    if not fracture_dir.is_dir() or not fracture_dir.name.startswith("fractured_"):
                        continue

                    obj_files = sorted(fracture_dir.glob("piece_*.obj"))
                    if len(obj_files) < 2:
                        continue

                    fid = make_id(piece_type, category_dir.name, hash_dir.name, fracture_dir.name)
                    out_path = OUTPUT_DIR / f"{fid}.png"

                    if out_path.exists():
                        skipped += 1
                        continue

                    render_thumbnail(obj_files, out_path)
                    generated += 1

                    if (generated + skipped) % 10 == 0:
                        print(f"  [{generated + skipped}] generated={generated}, skipped={skipped}")

    print(f"\nDone. Generated {generated} thumbnails, skipped {skipped} existing.")


if __name__ == "__main__":
    main()
