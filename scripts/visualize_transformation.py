import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


COLORS = {
    0: (0.0, 0.8, 0.0, 0.7),   # green
    1: (0.8, 0.0, 0.0, 0.7),   # red
}


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return mesh


def add_mesh_to_ax(ax, mesh, color):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    triangles = verts[faces]
    poly = Poly3DCollection(triangles, alpha=color[3])
    poly.set_facecolor(color[:3])
    poly.set_edgecolor((0, 0, 0, 0.05))
    ax.add_collection3d(poly)


def set_equal_axes(ax, meshes):
    all_verts = np.vstack([np.array(m.vertices) for m in meshes])
    center = all_verts.mean(axis=0)
    max_range = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2
    for i, fn in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        fn(center[i] - max_range, center[i] + max_range)


def visualize(save_dir="results/viz_output/wine_glass_2"):
    p0_assembled = load_mesh(f"{save_dir}/piece0_assembled.obj")
    p1_assembled = load_mesh(f"{save_dir}/piece1_assembled.obj")
    p0_gt = load_mesh(f"{save_dir}/piece0_ground_truth.obj")
    p1_gt = load_mesh(f"{save_dir}/piece1_ground_truth.obj")

    all_meshes = [p0_assembled, p1_assembled, p0_gt, p1_gt]

    fig = plt.figure(figsize=(20, 7))

    # predicted assembly
    ax1 = fig.add_subplot(131, projection='3d')
    add_mesh_to_ax(ax1, p0_assembled, COLORS[0])
    add_mesh_to_ax(ax1, p1_assembled, COLORS[1])
    set_equal_axes(ax1, all_meshes)
    ax1.set_title("Predicted Assembly", fontsize=14, fontweight='bold')

    # ground truth
    ax2 = fig.add_subplot(132, projection='3d')
    add_mesh_to_ax(ax2, p0_gt, COLORS[0])
    add_mesh_to_ax(ax2, p1_gt, COLORS[1])
    set_equal_axes(ax2, all_meshes)
    ax2.set_title("Ground Truth", fontsize=14, fontweight='bold')

    # overlay
    ax3 = fig.add_subplot(133, projection='3d')
    add_mesh_to_ax(ax3, p0_assembled, (*COLORS[0][:3], 0.5))
    add_mesh_to_ax(ax3, p1_assembled, (*COLORS[1][:3], 0.5))
    add_mesh_to_ax(ax3, p0_gt, (0.5, 0.5, 0.5, 0.2))
    add_mesh_to_ax(ax3, p1_gt, (0.5, 0.5, 0.5, 0.2))
    set_equal_axes(ax3, all_meshes)
    ax3.set_title("Overlay (color=predicted, gray=GT)", fontsize=14, fontweight='bold')

    # sync viewing angle across all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=135)

    plt.tight_layout()
    out_path = f"{save_dir}/assembly_visualization.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    visualize()
