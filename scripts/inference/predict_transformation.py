"""
Run inference on a folder of OBJ pieces and evaluate assembly quality.

Usage:
    python -m scripts.inference.predict_transformation \
        --cfg experiments/double_attn_scripts/everyday_eval.yaml \
        --pieces_dir /path/to/object_folder_with_objs \
        --save_dir results/viz_output \
        --seed 42
"""

import os
import argparse

import numpy as np
import torch
import trimesh
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn, TrainerState, RunningStage
from scipy.spatial.transform import Rotation as R

from utilities.utils_config import CONFIG, config_from_file
from jigsaw_pipeline import build_jigsaw_model
from jigsaw_pipeline.joint_segmentation_align_model import JointSegmentationAlignmentModel

PIECE_COLORS = [
    [0, 200, 0, 255],
    [200, 0, 0, 255],
    [0, 0, 200, 255],
    [200, 200, 0, 255],
]


def load_model(config):
    model = build_jigsaw_model(config)
    ckp = torch.load(config.WEIGHT_FILE, map_location="cpu", weights_only=False)
    if "state_dict" in ckp:
        model = JointSegmentationAlignmentModel.load_from_checkpoint(checkpoint_path=config.WEIGHT_FILE, strict=False, config=config)
    else:
        model.load_state_dict(ckp, strict=False)
    model.eval()
    model.cuda()

    trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False, enable_progress_bar=False)
    trainer.state = TrainerState(fn=TrainerFn.TESTING, stage=RunningStage.TESTING)
    model.trainer = trainer
    return model



def load_meshes_from_dir(pieces_dir: str) -> list[trimesh.Trimesh]:
    """Discover and load all OBJ files in a folder (sorted)."""
    files = sorted(f for f in os.listdir(pieces_dir) if f.endswith(".obj"))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 OBJ files in {pieces_dir}, found {len(files)}")

    meshes = []
    for fname in files:
        mesh = trimesh.load(os.path.join(pieces_dir, fname), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if isinstance(mesh, list):
            if len(mesh) == 0:
                raise ValueError(f"Corrupted mesh: {fname}")
            mesh = trimesh.util.concatenate(mesh)
        meshes.append(mesh)

    print(f"Loaded {len(meshes)} pieces from {pieces_dir}: {files}")
    return meshes


def distribute_points(areas: np.ndarray, total_points: int, min_points: int = 30):
    """Distribute sample budget across pieces proportionally to surface area."""
    total_area = np.sum(areas)
    nr_points = np.ceil(areas * total_points / total_area).astype(np.int32)

    diff = int(np.sum(nr_points) - total_points)
    nr_points[np.argmax(nr_points)] -= diff

    for i in range(len(nr_points)):
        if nr_points[i] < min_points:
            delta = min_points - nr_points[i]
            nr_points[i] = min_points
            largest = np.argmax(nr_points)
            if largest != i:
                nr_points[largest] -= delta

    return nr_points.astype(np.int64)


def pad(arr: np.ndarray, target_rows: int) -> np.ndarray:
    if arr.shape[0] >= target_rows:
        return arr
    pad_shape = (target_rows,) + arr.shape[1:]
    padded = np.zeros(pad_shape, dtype=arr.dtype)
    padded[: arr.shape[0]] = arr
    return padded


def preprocess_meshes(meshes: list[trimesh.Trimesh], config, seed: int):
    """
    Same pipeline as FracturePairsDataset / FractureAssemblyDataset:
      1. Area-proportional point sampling
      2. Center each piece at origin
      3. Random rotation (deterministic via seed)
      4. Shuffle point order

    Returns (data_dict, transforms) where transforms is a list of
    (centroid, rot_matrix) per piece, needed for post-processing.
    """
    np.random.seed(seed)

    num_pieces = len(meshes)
    num_points = config.DATA.NUM_PC_POINTS
    max_parts = config.DATA.MAX_NUM_PART
    fracture_threshold = config.DATA.FRACTURE_LABEL_THRESHOLD

    areas = np.array([m.area for m in meshes])
    pts_per_piece = distribute_points(areas, num_points)

    assembled_pcs = []
    gt_assembled_pcs = []
    gt_translations = []
    gt_rotations = []
    transforms = []

    for i, mesh in enumerate(meshes):
        n_pts = int(pts_per_piece[i])
        sampled, _ = mesh.sample(n_pts, return_index=True)
        gt_pc = sampled.copy()

        centroid = np.mean(sampled, axis=0)
        sampled = sampled - centroid

        rot_mat = R.random().as_matrix()
        sampled = (rot_mat @ sampled.T).T
        gt_quat = R.from_matrix(rot_mat.T).as_quat()          # scipy [x,y,z,w]
        gt_quat = gt_quat[[3, 0, 1, 2]].astype(np.float32)   # model [w,x,y,z]

        order = np.arange(n_pts)
        np.random.shuffle(order)
        sampled = sampled[order]
        gt_pc = gt_pc[order]

        assembled_pcs.append(sampled.astype(np.float32))
        gt_assembled_pcs.append(gt_pc.astype(np.float32))
        gt_translations.append(centroid.astype(np.float32))
        gt_rotations.append(gt_quat)
        transforms.append((centroid, rot_mat))

    assembled_pcs = np.concatenate(assembled_pcs)
    gt_assembled_pcs = np.concatenate(gt_assembled_pcs)
    gt_translations = pad(np.stack(gt_translations), max_parts)
    gt_rotations = pad(np.stack(gt_rotations), max_parts)
    points_per_part = pad(np.array(pts_per_piece, dtype=np.int64), max_parts)

    valids = np.zeros(max_parts, dtype=np.float32)
    valids[:num_pieces] = 1.0

    thresholds = np.full(num_points, fracture_threshold, dtype=np.float32)

    data_dict = {
        "part_pcs": torch.tensor(assembled_pcs).unsqueeze(0).cuda(),
        "gt_pcs": torch.tensor(gt_assembled_pcs).unsqueeze(0).cuda(),
        "part_valids": torch.tensor(valids).unsqueeze(0).cuda(),
        "part_quat": torch.tensor(gt_rotations).unsqueeze(0).cuda(),
        "part_trans": torch.tensor(gt_translations).unsqueeze(0).cuda(),
        "n_pcs": torch.tensor(points_per_part).unsqueeze(0).cuda(),
        "data_id": 0,
        "critical_label_thresholds": torch.tensor(thresholds).unsqueeze(0).cuda(),
    }

    return data_dict, transforms


def compute_errors(pred_transforms, data_dict):
    """Print and return per-piece rotation / translation errors vs GT."""
    gt_quats = data_dict["part_quat"][0].cpu().numpy()
    gt_trans = data_dict["part_trans"][0].cpu().numpy()
    n_valid = int(data_dict["part_valids"][0].sum().item())

    errors = []
    for i in range(n_valid):
        pred_R = pred_transforms["rot"][0, i]
        pred_t = pred_transforms["trans"][0, i]

        q = gt_quats[i] # gt rotation quaternion [w, x, y, z]
        gt_R = R.from_quat(q[[1, 2, 3, 0]]).as_matrix() # scipy [x,y,z,w]
        gt_t = gt_trans[i]

        R_diff = pred_R @ gt_R.T # angular error between predicted and GT rotation
        angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
        t_err = float(np.linalg.norm(pred_t - gt_t)) # translation error

        errors.append({"piece": i, "rot_err_deg": angle_deg, "trans_err": t_err})
        print(f"  Piece {i}: rotation error = {angle_deg:.2f}°, translation error = {t_err:.4f}")

    return errors


def assemble_and_save(meshes, transforms, pred_transforms, save_dir):
    """Apply predicted R,t to original meshes, save assembled + ground truth."""
    os.makedirs(save_dir, exist_ok=True)

    pred_rot = pred_transforms["rot"]
    pred_trans = pred_transforms["trans"]

    assembled_meshes = []
    gt_meshes = []

    for i, mesh in enumerate(meshes):
        centroid, rot_mat = transforms[i]
        verts = np.array(mesh.vertices)

        scrambled = (rot_mat @ (verts - centroid).T).T
        pred_R = pred_rot[0, i]
        pred_t = pred_trans[0, i]
        new_verts = (pred_R @ scrambled.T).T + pred_t

        color = PIECE_COLORS[i % len(PIECE_COLORS)]

        assembled_mesh = mesh.copy()
        assembled_mesh.vertices = new_verts
        assembled_mesh.visual.face_colors = color
        assembled_meshes.append(assembled_mesh)
        assembled_mesh.export(os.path.join(save_dir, f"piece{i}_assembled.obj"))

        gt_mesh = mesh.copy()
        gt_mesh.visual.face_colors = color
        gt_meshes.append(gt_mesh)
        gt_mesh.export(os.path.join(save_dir, f"piece{i}_ground_truth.obj"))

    combined_assembled = trimesh.util.concatenate(assembled_meshes)
    combined_assembled.export(os.path.join(save_dir, "assembled.obj"))

    combined_gt = trimesh.util.concatenate(gt_meshes)
    combined_gt.export(os.path.join(save_dir, "ground_truth.obj"))

    piece_names = ", ".join(f"piece{i}" for i in range(len(meshes)))
    print(f"\nSaved to {save_dir}/")
    print(f"  assembled.obj         - all pieces in predicted positions")
    print(f"  ground_truth.obj      - all pieces in original positions")
    print(f"  {{{piece_names}}}_assembled.obj")
    print(f"  {{{piece_names}}}_ground_truth.obj")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jigsaw inference on a folder of OBJ pieces")
    parser.add_argument("--cfg", required=True, help="Path to eval YAML config")
    parser.add_argument("--pieces_dir", required=True, help="Folder containing the OBJ pieces (auto-discovers all .obj files)")
    parser.add_argument("--save_dir", default="results/viz_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_from_file(args.cfg)
    torch.manual_seed(args.seed)

    meshes = load_meshes_from_dir(args.pieces_dir)
    model = load_model(CONFIG)
    data_dict, transforms = preprocess_meshes(meshes, CONFIG, args.seed)

    with torch.no_grad():
        out_dict = model.forward(data_dict)
    pred_transforms = model.global_alignment(data_dict, out_dict)

    print("=" * 50)
    errors = compute_errors(pred_transforms, data_dict)
    print("=" * 50)

    assemble_and_save(meshes, transforms, pred_transforms, args.save_dir)
