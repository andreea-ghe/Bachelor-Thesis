#!/usr/bin/env python3
"""
Usage:
    python -m scripts.visualization \
        --cfg experiments/double_attn_scripts/everyday_eval.yaml \
        --piece1 /path/to/piece_0.obj \
        --piece2 /path/to/piece_1.obj \
        --save_dir results/viz_output
"""
import os
import torch
import numpy as np
import argparse
import open3d as o3d

from utilities.utils_config import CONFIG, config_from_file
from jigsaw_pipeline import build_jigsaw_model
from jigsaw_pipeline.joint_segmentation_align_model import JointSegmentationAlignmentModel
from dataset_preprocessing.fracture_pairs_dataset import FracturePairsDataset


COLORS = [[0.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.8], [0.8, 0.8, 0.0]]


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


def prepare_pair(piece1_path, piece2_path, config):
    dataset = FracturePairsDataset.__new__(FracturePairsDataset)
    dataset.dataset_dir = ""
    dataset.num_points = config.DATA.NUM_PC_POINTS
    dataset.rot_range = config.DATA.ROT_RANGE
    dataset.fracture_label_threshold = config.DATA.FRACTURE_LABEL_THRESHOLD
    dataset.max_parts = 2
    dataset.min_num_points = 30
    dataset.pairs = [(piece1_path, piece2_path)]
    dataset.length = 1

    sample = dataset[0]
    data_dict = {}
    for k, v in sample.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            data_dict[k] = torch.as_tensor(v).unsqueeze(0).cuda()
        else:
            data_dict[k] = v
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--piece1", required=True)
    parser.add_argument("--piece2", required=True)
    parser.add_argument("--save_dir", type=str, default="results/viz_output")
    args = parser.parse_args()

    config_from_file(args.cfg)
    torch.manual_seed(CONFIG.RANDOM_SEED)

    model = load_model(CONFIG)
    data_dict = prepare_pair(args.piece1, args.piece2, CONFIG)

    with torch.no_grad():
        out_dict = model.forward(data_dict)
    pred_transforms = model.global_alignment(data_dict, out_dict)

    # apply predicted R, t to input pieces and save
    n_pcs = data_dict['n_pcs'][0].cpu().numpy().astype(int)
    part_pcs = data_dict['part_pcs'][0].cpu().numpy()[:, :3]

    os.makedirs(args.save_dir, exist_ok=True)
    n_valid = int(data_dict['part_valids'][0].sum().item())
    offset = 0

    for i in range(n_valid):
        pts = part_pcs[offset:offset + n_pcs[i]]
        R = pred_transforms['rot'][0, i]
        t = pred_transforms['trans'][0, i]
        assembled = (R @ pts.T).T + t

        print(f"Piece {i}: R =\n{R}\n  t = {t}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(assembled)
        pcd.paint_uniform_color(COLORS[i % len(COLORS)])
        o3d.io.write_point_cloud(os.path.join(args.save_dir, f"piece{i}_assembled.ply"), pcd)
        offset += n_pcs[i]

    # ground truth for comparison
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(data_dict['gt_pcs'][0].cpu().numpy()[:, :3])
    gt_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud(os.path.join(args.save_dir, "ground_truth.ply"), gt_pcd)

    print(f"\nSaved to {args.save_dir}/")
