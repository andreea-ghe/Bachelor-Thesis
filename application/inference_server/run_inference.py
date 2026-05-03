import os
import sys

import numpy as np
import torch
import trimesh
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn, TrainerState, RunningStage
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utilities.utils_config import CONFIG, config_from_file
from jigsaw_pipeline import build_jigsaw_model
from jigsaw_pipeline.joint_segmentation_align_model import JointSegmentationAlignmentModel

SEED = 42
PIECE_COLORS = [
    [0, 200, 0, 255],
    [200, 0, 0, 255],
    [0, 0, 200, 255],
    [200, 200, 0, 255],
]

_model_cache: dict = {}


def _get_model(config_path: str):
    """Load a model from a YAML config, caching it for subsequent requests."""
    if config_path in _model_cache:
        return _model_cache[config_path]

    config_from_file(config_path)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = build_jigsaw_model(CONFIG)
    ckp = torch.load(CONFIG.WEIGHT_FILE, map_location="cpu", weights_only=False)
    if "state_dict" in ckp:
        model = JointSegmentationAlignmentModel.load_from_checkpoint(checkpoint_path=CONFIG.WEIGHT_FILE, strict=False, config=CONFIG,)
    else:
        model.load_state_dict(ckp, strict=False)

    model.eval()
    model.cuda()

    trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False, enable_progress_bar=False)
    trainer.state = TrainerState(fn=TrainerFn.TESTING, stage=RunningStage.TESTING)
    model.trainer = trainer

    import copy
    _model_cache[config_path] = (model, copy.deepcopy(CONFIG))
    return _model_cache[config_path]


# Preprocessing (mirrors FracturePairsDataset / FractureAssemblyDataset)
def _distribute_points(areas: np.ndarray, total_points: int, min_points: int = 30):
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


def _preprocess_meshes(meshes: list[trimesh.Trimesh], config):
    """
    Reproduce the preprocessing applied to the training datasets:
    - sample points from each mesh proportionally to surface area
    - center each piece at origin  (centroid = gt_trans)
    - apply a deterministic random rotation  (inverse = gt_quat)
    - shuffle point order

    Output:
        data_dict : dict ready to be batched and sent to the model
        transforms : list of per-piece (centroid, rot_matrix) so we can later
                       apply the same preprocessing to the full mesh vertices
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    num_pieces = len(meshes)
    num_points = config.DATA.NUM_PC_POINTS
    max_parts = config.DATA.MAX_NUM_PART
    fracture_threshold = config.DATA.FRACTURE_LABEL_THRESHOLD

    areas = np.array([m.area for m in meshes])
    pts_per_piece = _distribute_points(areas, num_points)

    assembled_pcs = []
    gt_assembled_pcs = []
    gt_translations = []
    gt_rotations = []
    transforms = []

    for i, mesh in enumerate(meshes):
        n_pts = int(pts_per_piece[i])
        sampled, _ = mesh.sample(n_pts, return_index=True)
        gt_pc = sampled.copy()

        # center at origin
        centroid = np.mean(sampled, axis=0)
        sampled = sampled - centroid

        # deterministic random rotation (seeded above)
        rot_mat = R.random().as_matrix()
        sampled = (rot_mat @ sampled.T).T
        gt_quat = R.from_matrix(rot_mat.T).as_quat() # scipy: [x,y,z,w]
        gt_quat = gt_quat[[3, 0, 1, 2]].astype(np.float32) # model wants [w,x,y,z]

        # shuffle point order
        order = np.arange(n_pts)
        np.random.shuffle(order)
        sampled = sampled[order]
        gt_pc = gt_pc[order]

        assembled_pcs.append(sampled.astype(np.float32))
        gt_assembled_pcs.append(gt_pc.astype(np.float32))
        gt_translations.append(centroid.astype(np.float32))
        gt_rotations.append(gt_quat)
        transforms.append((centroid, rot_mat))

    # concatenate
    assembled_pcs = np.concatenate(assembled_pcs) # [N_total, 3]
    gt_assembled_pcs = np.concatenate(gt_assembled_pcs) # [N_total, 3]
    gt_translations = np.stack(gt_translations) # [P, 3]
    gt_rotations = np.stack(gt_rotations) # [P, 4]
    points_per_part = np.array(pts_per_piece, dtype=np.int64)

    # pad to max_parts if needed (multi-piece models may have max_parts=4)
    def _pad(arr, target_rows):
        if arr.shape[0] >= target_rows:
            return arr
        pad_shape = (target_rows,) + arr.shape[1:]
        padded = np.zeros(pad_shape, dtype=arr.dtype)
        padded[: arr.shape[0]] = arr
        return padded

    gt_translations = _pad(gt_translations, max_parts)
    gt_rotations = _pad(gt_rotations, max_parts)
    points_per_part = _pad(points_per_part, max_parts)

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


# apply predicted transforms to the original meshes
def _assemble_meshes(
    meshes: list[trimesh.Trimesh],
    transforms: list[tuple[np.ndarray, np.ndarray]],
    pred_rot: np.ndarray,
    pred_trans: np.ndarray,
) -> list[trimesh.Trimesh]:
    """
    For each piece, apply the same centering + rotation used during
    preprocessing, then apply the model's predicted R, t.

    Returns a list of assembled trimesh objects.
    """
    assembled = []
    for i, mesh in enumerate(meshes):
        centroid, rot_mat = transforms[i]
        verts = np.array(mesh.vertices)

        # same preprocessing: center then rotate
        scrambled = (rot_mat @ (verts - centroid).T).T

        # predicted assembly transform
        pred_R = pred_rot[0, i]   # [3, 3]
        pred_t = pred_trans[0, i] # [3]
        new_verts = (pred_R @ scrambled.T).T + pred_t

        out = mesh.copy()
        out.vertices = new_verts
        out.visual.face_colors = PIECE_COLORS[i % len(PIECE_COLORS)]
        assembled.append(out)

    return assembled


def run_inference(config_path: str, obj_strings: list[str]) -> dict:
    """
    Main entry point called by the FastAPI server.

    Input:
        config_path : str absolute path to a YAML eval config.
        obj_strings : list[str] OBJ file contents for each piece (2-4 pieces).

    Output:
        dict with:
            pieces : list of dicts, each with ``name``, ``assembled_obj``
            combined_assembled_obj : str (OBJ of all pieces together)
    """
    meshes = []
    for i, obj_text in enumerate(obj_strings):
        mesh = trimesh.load(trimesh.util.wrap_as_stream(obj_text), file_type="obj", force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        meshes.append(mesh)

    model, config = _get_model(config_path)
    data_dict, transforms = _preprocess_meshes(meshes, config)

    with torch.no_grad():
        out_dict = model.forward(data_dict)
    pred_transforms = model.global_alignment(data_dict, out_dict)

    pred_rot = pred_transforms["rot"] # [1, P, 3, 3]
    pred_trans = pred_transforms["trans"] # [1, P, 3]

    assembled_meshes = _assemble_meshes(meshes, transforms, pred_rot, pred_trans)

    pieces = []
    for i, mesh in enumerate(assembled_meshes):
        pieces.append({
            "name": f"piece_{i}",
            "assembled_obj": mesh.export(file_type="obj"),
        })

    combined = trimesh.util.concatenate(assembled_meshes)

    return {
        "pieces": pieces,
        "combined_assembled_obj": combined.export(file_type="obj"),
    }
