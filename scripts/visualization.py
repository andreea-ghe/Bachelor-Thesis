#!/usr/bin/env python3
"""
Visualize Jigsaw assembly results.

Loads a trained model, runs inference on a test sample,
recovers the predicted transformation (R, t) for each piece,
and visualizes the original vs assembled point clouds.

Usage:
    python -m scripts.visualization --cfg experiments/new_arch_scripts/everyday_eval.yaml --sample_idx 50
"""
import os
import sys
import torch
import numpy as np
import argparse
import open3d as o3d

from utilities.utils_config import CONFIG
from dataset_preprocessing import build_pairs_test_loader
from jigsaw_pipeline import build_jigsaw_model
from jigsaw_pipeline.joint_segmentation_align_model import JointSegmentationAlignmentModel


COLORS = [
    [0.0, 0.8, 0.0],   # green  (piece 0)
    [0.8, 0.0, 0.0],   # red    (piece 1)
    [0.0, 0.0, 0.8],   # blue   (piece 2)
    [0.8, 0.8, 0.0],   # yellow (piece 3)
]


def load_model(config):
    model = build_jigsaw_model(config)

    ckp = torch.load(config.WEIGHT_FILE, map_location='cpu', weights_only=False)
    if 'state_dict' in ckp:
        model = JointSegmentationAlignmentModel.load_from_checkpoint(
            checkpoint_path=config.WEIGHT_FILE, strict=False, config=config
        )
    else:
        model.load_state_dict(ckp, strict=False)

    model.eval()
    model.cuda()
    return model


def get_sample(config, sample_idx):
    test_loader = build_pairs_test_loader(config)
    dataset = test_loader.dataset

    if sample_idx >= len(dataset):
        print(f"Sample index {sample_idx} out of range (dataset has {len(dataset)} samples)")
        sys.exit(1)

    sample = dataset[sample_idx]
    # add batch dimension
    data_dict = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.unsqueeze(0).cuda()
        else:
            data_dict[k] = v
    return data_dict


def run_inference(model, data_dict):
    with torch.no_grad():
        out_dict = model.forward(data_dict)

    pred_transforms = model.global_alignment(data_dict, out_dict)
    return out_dict, pred_transforms


def extract_piece_points(data_dict, piece_idx):
    """Extract XYZ points for a single piece from the concatenated point cloud."""
    n_pcs = data_dict['n_pcs'][0].cpu().numpy().astype(int)  # [P]
    part_pcs = data_dict['part_pcs'][0].cpu().numpy()  # [N_SUM, 3+]

    start = int(np.sum(n_pcs[:piece_idx]))
    end = start + int(n_pcs[piece_idx])
    return part_pcs[start:end, :3]


def build_point_clouds(data_dict, pred_transforms, piece_idx):
    """Build Open3D point clouds for a single piece: input and assembled."""
    piece_points = extract_piece_points(data_dict, piece_idx)

    # pred_transforms are numpy arrays from global_alignment
    R = pred_transforms['rot'][0, piece_idx]   # [3, 3]
    t = pred_transforms['trans'][0, piece_idx]  # [3]
    assembled_points = (R @ piece_points.T).T + t

    color = COLORS[piece_idx % len(COLORS)]

    input_pcd = o3d.geometry.PointCloud()
    input_pcd.points = o3d.utility.Vector3dVector(piece_points)
    input_pcd.paint_uniform_color(color)

    assembled_pcd = o3d.geometry.PointCloud()
    assembled_pcd.points = o3d.utility.Vector3dVector(assembled_points)
    assembled_pcd.paint_uniform_color(color)

    return input_pcd, assembled_pcd


def build_gt_point_cloud(data_dict):
    """Build GT point cloud — gt_pcs is already in the correct assembled position."""
    gt_pcs = data_dict['gt_pcs'][0].cpu().numpy()[:, :3]
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pcs)
    gt_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return gt_pcd


def visualize(data_dict, pred_transforms):
    n_valid = int(data_dict['part_valids'][0].sum().item())

    input_pcds = []
    assembled_pcds = []
    for i in range(n_valid):
        inp, assem = build_point_clouds(data_dict, pred_transforms, i)
        input_pcds.append(inp)
        assembled_pcds.append(assem)

    gt_pcd = build_gt_point_cloud(data_dict)

    # shift assembled view to the right for side-by-side comparison
    offset = np.array([2.0, 0.0, 0.0])
    for pcd in assembled_pcds:
        pcd.translate(offset)
    gt_shifted = build_gt_point_cloud(data_dict)
    gt_shifted.translate(offset * 2)

    all_geoms = input_pcds + assembled_pcds + [gt_shifted]

    print(f"\nVisualization: LEFT = input pieces | MIDDLE = predicted assembly | RIGHT = ground truth")
    print(f"Pieces: {n_valid}")
    for i in range(n_valid):
        R = pred_transforms['rot'][0, i]
        t = pred_transforms['trans'][0, i]
        print(f"  Piece {i}: t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

    o3d.visualization.draw_geometries(all_geoms, window_name="Jigsaw Assembly Visualization")


def save_point_clouds(data_dict, pred_transforms, output_dir, sample_idx):
    """Save point clouds as PLY files for offline viewing."""
    os.makedirs(output_dir, exist_ok=True)
    n_valid = int(data_dict['part_valids'][0].sum().item())

    for i in range(n_valid):
        inp, assem = build_point_clouds(data_dict, pred_transforms, i)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"sample{sample_idx}_piece{i}_input.ply"), inp)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"sample{sample_idx}_piece{i}_assembled.ply"), assem)

    gt_pcd = build_gt_point_cloud(data_dict)
    o3d.io.write_point_cloud(os.path.join(output_dir, f"sample{sample_idx}_ground_truth.ply"), gt_pcd)
    print(f"Saved point clouds to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to eval config YAML")
    parser.add_argument("--sample_idx", type=int, default=50, help="Test sample index")
    parser.add_argument("--save_dir", type=str, default=None, help="Save PLY files instead of visualizing")
    args = parser.parse_args()

    # load config
    CONFIG.merge_from_file(args.cfg)
    torch.manual_seed(CONFIG.RANDOM_SEED)

    print(f"Loading model from: {CONFIG.WEIGHT_FILE}")
    model = load_model(CONFIG)

    print(f"Loading sample {args.sample_idx} from test set...")
    data_dict = get_sample(CONFIG, args.sample_idx)

    print("Running inference...")
    out_dict, pred_transforms = run_inference(model, data_dict)

    if args.save_dir:
        save_point_clouds(data_dict, pred_transforms, args.save_dir, args.sample_idx)
    else:
        visualize(data_dict, pred_transforms)
