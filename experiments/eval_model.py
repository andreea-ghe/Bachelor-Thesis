import os
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from datetime import datetime
from dataset_preprocessing.fracture_assembly_dataset import build_test_loader
from utilities.utils_stdout import DuplicateStdoutFileManager
from utilities.utils_parse_args import parse_args
from utilities.utils_config import CONFIG
from utilities.utils_edict import print_edict
from jigsaw_pipeline import build_jigsaw_model
from jigsaw_pipeline.joint_segmentation_align_model import JointSegmentationAlignmentModel
from pytorch_lightning.loggers import CSVLogger


NOW_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def test_model(config):
    """
    Evaluation function for the jigsaw model.
    Tests the trained model on the validation/test dataset and computes metrics:
    - Chamfer Distance (CD): overall alignment quality
    - Part Accuracy (PA): how many pieces matched correctly
    - Rotation Error (RE): accuracy of predicted rotations
    - Translation Error (TE): accuracy of predicted translations

    Uses predicted fracture segmentation (not ground truth),
    Hungarian algorithm for discrete matching, and
    global alignment to recover all poses.

    Input:
        config: configuration object with test settings
    """
    if len(config.STATS):
        os.makedirs(config.STATS, exist_ok=True) # create stats directory if needed

    test_loader = build_test_loader(config)
    model = build_jigsaw_model(config)


    # setup logging
    model_save_path = config.MODEL_SAVE_PATH # for model checkpoints

    if config.LOG_FILE_NAME is not None and len(config.LOG_FILE_NAME) > 0:
        logger_name = f"{config.MODEL_NAME}_{config.LOG_FILE_NAME}"
    else:
        logger_name = f"{config.MODEL_NAME}_{NOW_TIME}"
    logger = CSVLogger(
        save_dir=model_save_path,
        name=logger_name,
    )

    callbacks = []

    all_gpus = list(config.GPUS)
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=all_gpus,
        strategy="ddp" if len(all_gpus) > 1 else "auto",
        callbacks=callbacks,
    )

    ckp_files = os.listdir(model_save_path)
    ckp_files = [ckp for ckp in ckp_files if "model_" in ckp]

    weights_already_loaded = False

    if config.WEIGHT_FILE:
        ckp = torch.load(config.WEIGHT_FILE, map_location='cpu', weights_only=False)

        if 'state_dict' in ckp:
            ckp_path = config.WEIGHT_FILE
            ckp_keys = set(ckp['state_dict'].keys())
            model_keys = set(model.state_dict().keys())
            missing_in_ckp = model_keys - ckp_keys
            missing_in_model = ckp_keys - model_keys
            if missing_in_ckp:
                print(f"Keys in model but NOT in checkpoint ({len(missing_in_ckp)}):")
                for k in sorted(missing_in_ckp)[:10]:
                    print(f"  {k}")
            if missing_in_model:
                print(f"Keys in checkpoint but NOT in model ({len(missing_in_model)}):")
                for k in sorted(missing_in_model)[:10]:
                    print(f"  {k}")
            if not missing_in_ckp and not missing_in_model:
                print("All checkpoint keys match model keys!")
        else:
            result = model.load_state_dict(ckp, strict=False)
            if result.missing_keys:
                print(f"WARNING: {len(result.missing_keys)} missing keys when loading weights")
            if result.unexpected_keys:
                print(f"WARNING: {len(result.unexpected_keys)} unexpected keys when loading weights")
            if not result.missing_keys and not result.unexpected_keys:
                print("All weights loaded successfully!")
            weights_already_loaded = True
            ckp_path = None

    elif ckp_files: # auto detect latest checkpoint
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x)),
        )
        last_ckp = ckp_files[-1]
        ckp_path = os.path.join(model_save_path, last_ckp)
    else: # no checkpoint found
        ckp_path = None

    
    if not weights_already_loaded and ckp_path is not None:
        model = JointSegmentationAlignmentModel.load_from_checkpoint(checkpoint_path=ckp_path, strict=False, config=config)
    elif not weights_already_loaded:
        print("WARNING: No checkpoint found — evaluating with random weights!")
    
    print("Finish Setting -----")
    trainer.test(model, test_loader)

    print("Done evaluation")


if __name__ == "__main__":
    """
    Usage:
        python eval_model.py --cfg experiments/jigsaw_250e_cosine.yaml
        python eval_model.py --cfg experiments/jigsaw_250e_cosine.yaml --weight path/to/checkpoint.ckpt
    """
    args = parse_args("Jigsaw")
    torch.manual_seed(CONFIG.RANDOM_SEED)

    # setup evaluation log file
    file_end = NOW_TIME
    if CONFIG.LOG_FILE_NAME is not None and len(CONFIG.LOG_FILE_NAME) > 0:
        file_end += "_{}".format(CONFIG.LOG_FILE_NAME)
    full_log_name = f"eval_log_{file_end}"

    # duplicate stdout to log file
    with DuplicateStdoutFileManager(os.path.join(CONFIG.OUTPUT_PATH, f"{full_log_name}.log")) as _:
        print_edict(CONFIG)  
        test_model(CONFIG)
