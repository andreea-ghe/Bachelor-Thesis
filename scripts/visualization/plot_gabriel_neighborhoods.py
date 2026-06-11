import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from torch_geometric.nn.pool import fps, knn
from torch_geometric.utils import to_dense_batch
from feature_extractor.utils_encoder_decoder import gabriel_filter
from feature_extractor.utils import select_points


def sample_points(mesh: trimesh.Trimesh, num_points: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    pts, _ = mesh.sample(num_points, return_index=True)
    return pts.astype(np.float32)


def compute_neighborhoods(
    points: np.ndarray, fps_ratio: float, K: int, gabriel_ratio: float | None
):
    """
    Run FPS -> kNN -> optional Gabriel filter.
    Returns (centroids_xyz, all_points, kept_edges) where kept_edges is a
    list of (centroid_idx, [(nb_x, nb_y, nb_z), ...]).
    """
    xyz = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # [1, N, 3]
    batch = torch.zeros(points.shape[0], dtype=torch.long)

    centroid_idx = fps(xyz[0], batch=batch, ratio=fps_ratio).unsqueeze(0)
    centroids_xyz = select_points(xyz, centroid_idx)
    S = centroids_xyz.shape[1]

    centroid_batch = torch.zeros(S, dtype=torch.long)
    nb_raw = knn(xyz[0], centroids_xyz[0], k=K, batch_x=batch, batch_y=centroid_batch)
    neighborhood_idx = to_dense_batch(
        nb_raw[1], nb_raw[0], fill_value=-1, max_num_nodes=K
    )[0].unsqueeze(0)

    if gabriel_ratio is not None:
        valid_mask = neighborhood_idx != -1
        safe_idx = neighborhood_idx.clone()
        safe_idx[~valid_mask] = 0
        nb_xyz = select_points(xyz, safe_idx)
        neighborhood_idx = gabriel_filter(
            centroids_xyz, nb_xyz, neighborhood_idx, min_keep_ratio=gabriel_ratio
        )

    c_np = centroids_xyz[0].numpy()
    idx_np = neighborhood_idx[0].numpy()
    pts_np = xyz[0].numpy()

    edges = []
    for s in range(S):
        cx, cy, cz = c_np[s]
        for k in range(K):
            ni = idx_np[s, k]
            if ni == -1:
                continue
            nx, ny, nz = pts_np[ni]
            edges.append(((cx, cy, cz), (nx, ny, nz)))

    return c_np, pts_np, edges, idx_np


def count_stats(idx_np):
    valid = idx_np != -1
    avg_kept = valid.sum(axis=1).mean()
    total_valid = valid.sum()
    return avg_kept, total_valid


def plot_panel(ax, points, centroids, edges, title, K):
    _, total = count_stats(
        np.zeros((centroids.shape[0], K), dtype=np.int64)
    )  # dummy
    avg_kept = len(edges) / max(centroids.shape[0], 1)

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c='#e0e0e0', s=0.3, alpha=0.3, depthshade=True
    )

    if edges:
        lc = Line3DCollection(edges, colors='#4a90d9', linewidths=0.3, alpha=0.5)
        ax.add_collection3d(lc)

    ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c='#d9534f', s=8, zorder=5, depthshade=True
    )

    ax.set_title(f"{title}\n{len(edges)} edges, ~{avg_kept:.1f}/centroid",
                 fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str,
                        default="application/web/core/static/core/meshes/"
                                "two_pieces/Bowl/594b22f21daf33ce6aea2f18ee404fd5/"
                                "fractured_32/piece_0.obj")
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--fps_ratio", type=float, default=0.15)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.is_absolute():
        mesh_path = REPO_ROOT / mesh_path
    mesh = trimesh.load(str(mesh_path), force="mesh")
    print(f"Loaded mesh: {mesh_path.name} ({len(mesh.vertices)} vertices)")

    points = sample_points(mesh, args.num_points, args.seed)
    print(f"Sampled {args.num_points} points")

    configs = [
        ("kNN (no filtering)", None),
        ("Gabriel r=0\n(no protection)", 0.0),
        ("Gabriel r=0.50\n(50% retained)", 0.5),
        ("Gabriel r=0.75\n(75% retained)", 0.75),
    ]

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    all_centroids, all_edges, all_idx = [], [], []

    for i, (label, ratio) in enumerate(configs):
        torch.manual_seed(args.seed)
        c, pts, edges, idx_np = compute_neighborhoods(
            points, args.fps_ratio, args.K, ratio
        )
        all_centroids.append(c)
        all_edges.append(edges)
        all_idx.append(idx_np)

    # Shared camera angle
    for i, (label, ratio) in enumerate(configs):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        plot_panel(ax, points, all_centroids[i], all_edges[i], label, args.K)
        ax.view_init(elev=25, azim=135)

    category = mesh_path.parent.parent.parent.name
    fracture = mesh_path.parent.name
    fig.suptitle(
        f"Gabriel Graph Filtering — {category} / {fracture}\n"
        f"({args.num_points} points, FPS ratio={args.fps_ratio}, K={args.K})",
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    if args.output:
        out = Path(args.output)
    else:
        out = REPO_ROOT / "results" / "plots" / "gabriel_neighborhood_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
