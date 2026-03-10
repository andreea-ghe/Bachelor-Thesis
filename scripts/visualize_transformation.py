import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData


def load_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    xyz = np.column_stack([v['x'], v['y'], v['z']])
    rgb = np.column_stack([v['red'], v['green'], v['blue']]) / 255.0
    return xyz, rgb


def visualize_pieces(piece0_path, piece1_path, gt_path):
    p0, c0 = load_ply(piece0_path)
    p1, c1 = load_ply(piece1_path)
    gt, cg = load_ply(gt_path)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(p0[:, 0], p0[:, 1], p0[:, 2], c=c0, s=1)
    ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c=c1, s=1)
    ax1.set_title("Predicted Assembly")

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c=cg, s=1)
    ax2.set_title("Ground Truth")

    # overlay: predicted + GT
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(p0[:, 0], p0[:, 1], p0[:, 2], c=c0, s=1, alpha=0.6)
    ax3.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c=c1, s=1, alpha=0.6)
    ax3.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='gray', s=1, alpha=0.3)
    ax3.set_title("Overlay (predicted + GT)")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig("results/viz_output/assembly_visualization.png", dpi=200)
    print("Saved results/viz_output/assembly_visualization.png")
    plt.show()


if __name__ == "__main__":
    visualize_pieces(
        "results/viz_output/piece0_assembled.ply",
        "results/viz_output/piece1_assembled.ply",
        "results/viz_output/ground_truth.ply",
    )
