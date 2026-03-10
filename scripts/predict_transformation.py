#!/usr/bin/env python3
"""
Usage:
    python -m scripts.predict_transformation \
        --cfg experiments/double_attn_scripts/everyday_eval.yaml \
        --piece1 /path/to/piece_0.obj \
        --piece2 /path/to/piece_1.obj \
        --save_dir results/viz_output
"""
import os
import torch
import trimesh
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from utilities.utils_config import CONFIG, config_from_file
from jigsaw_pipeline import build_jigsaw_model
from jigsaw_pipeline.joint_segmentation_align_model import JointSegmentationAlignmentModel
from dataset_preprocessing.fracture_pairs_dataset import FracturePairsDataset


COLORS = [
    [0, 200, 0, 255],     # green
    [200, 0, 0, 255],     # red
    [0, 0, 200, 255],     # blue
    [200, 200, 0, 255],   # yellow
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


def compute_errors(pred_transforms, data_dict):
    gt_quats = data_dict['part_quat'][0].cpu().numpy()   # [P, 4] wxyz
    gt_trans = data_dict['part_trans'][0].cpu().numpy()   # [P, 3]
    n_valid = int(data_dict['part_valids'][0].sum().item())

    for i in range(n_valid):
        pred_R = pred_transforms['rot'][0, i]
        pred_t = pred_transforms['trans'][0, i]

        # GT rotation: quaternion [w,x,y,z] -> rotation matrix
        q = gt_quats[i]
        gt_R = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()  # scipy wants [x,y,z,w]
        gt_t = gt_trans[i]

        # angular error between predicted and GT rotation
        R_diff = pred_R @ gt_R.T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        angle_deg = np.degrees(angle)

        # translation error
        t_err = np.linalg.norm(pred_t - gt_t)

        print(f"Piece {i}:")
        print(f"  Rotation error: {angle_deg:.2f} degrees")
        print(f"  Translation error: {t_err:.4f}")
        print(f"  Pred t = {pred_t}")
        print(f"  GT   t = {gt_t}")


def save_transformed_meshes(piece_paths, pred_transforms, data_dict, save_dir):
    """Load original meshes, apply the same preprocessing + predicted R,t, save as .obj."""
    os.makedirs(save_dir, exist_ok=True)
    n_valid = int(data_dict['part_valids'][0].sum().item())

    gt_quats = data_dict['part_quat'][0].cpu().numpy()
    gt_trans = data_dict['part_trans'][0].cpu().numpy()

    assembled_meshes = []
    gt_meshes = []

    for i in range(n_valid):
        mesh = trimesh.load(piece_paths[i], force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        # the dataset recenters each piece, then rotates it
        # GT transform undoes this: gt_R @ (rotated_point) + gt_t = original_point
        # predicted transform: pred_R @ (rotated_point) + pred_t = assembled_point
        # to apply pred transform to the mesh, we need to:
        #   1) recenter the mesh (same as dataset)
        #   2) rotate (same random rotation as dataset applied)
        #   3) apply predicted R, t
        # but we don't have the exact random rotation used...
        # instead: apply GT inverse to go to scrambled space, then apply predicted R, t
        # scrambled = gt_R_inv @ (original - gt_t)
        # assembled = pred_R @ scrambled + pred_t

        q = gt_quats[i]
        gt_R = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        gt_t = gt_trans[i]

        pred_R = pred_transforms['rot'][0, i]
        pred_t = pred_transforms['trans'][0, i]

        # original mesh → scrambled space → predicted assembled space
        verts = np.array(mesh.vertices)
        scrambled = (np.linalg.inv(gt_R) @ (verts - gt_t).T).T
        assembled = (pred_R @ scrambled.T).T + pred_t

        # assembled mesh
        assembled_mesh = mesh.copy()
        assembled_mesh.vertices = assembled
        assembled_mesh.visual.face_colors = COLORS[i % len(COLORS)]
        assembled_meshes.append(assembled_mesh)
        assembled_mesh.export(os.path.join(save_dir, f"piece{i}_assembled.obj"))

        # ground truth mesh (original positions)
        gt_mesh = mesh.copy()
        gt_mesh.visual.face_colors = COLORS[i % len(COLORS)]
        gt_meshes.append(gt_mesh)
        gt_mesh.export(os.path.join(save_dir, f"piece{i}_ground_truth.obj"))

    # combined files
    combined_assembled = trimesh.util.concatenate(assembled_meshes)
    combined_assembled.export(os.path.join(save_dir, "assembled.obj"))

    combined_gt = trimesh.util.concatenate(gt_meshes)
    combined_gt.export(os.path.join(save_dir, "ground_truth.obj"))

    print(f"\nSaved to {save_dir}/")
    print(f"  assembled.obj        - both pieces in predicted positions")
    print(f"  ground_truth.obj     - both pieces in original positions")
    print(f"  piece{{0,1}}_assembled.obj   - individual predicted pieces")
    print(f"  piece{{0,1}}_ground_truth.obj - individual GT pieces")


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

    # attach a minimal trainer so model.forward() can check self.trainer.testing
    import pytorch_lightning as pl
    from pytorch_lightning.trainer.states import TrainerFn, TrainerState, RunningStage
    trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False, enable_progress_bar=False)
    trainer.state = TrainerState(fn=TrainerFn.TESTING, stage=RunningStage.TESTING)
    model.trainer = trainer

    with torch.no_grad():
        out_dict = model.forward(data_dict)
    pred_transforms = model.global_alignment(data_dict, out_dict)

    print("=" * 50)
    compute_errors(pred_transforms, data_dict)
    print("=" * 50)

    save_transformed_meshes(
        [args.piece1, args.piece2],
        pred_transforms, data_dict, args.save_dir,
    )
