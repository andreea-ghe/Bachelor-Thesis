import open3d as o3d

def visualize_pieces(piece1, piece2, gt):
    pcd0 = o3d.io.read_point_cloud(piece1)
    pcd1 = o3d.io.read_point_cloud(piece2)
    gt_pcd = o3d.io.read_point_cloud(gt)
    o3d.visualization.draw_geometries([pcd0, pcd1, gt_pcd])


if __name__ == "__main__":
    visualize_pieces("results/viz_output/piece0_assembled.ply", "results/viz_output/piece1_assembled.ply", "results/viz_output/ground_truth.ply")
    