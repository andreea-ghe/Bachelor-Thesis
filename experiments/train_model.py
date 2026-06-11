import os 
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from jigsaw_pipeline import build_jigsaw_model
from dataset_preprocessing import build_data_loaders
from datetime import datetime
from utilities.utils_stdout import DuplicateStdoutFileManager
from utilities.utils_parse_args import parse_args
from utilities.utils_config import CONFIG
from utilities.utils_edict import print_edict


NOW_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def train_model(config):
    """
    Training function for the Jigsaw model.
    Pipeline:
    1. initialize data loaders with dataset
    2. build the Jigsaw model
    3. configure PyTorch Lightning trainer with callbacks
    4. train with loss scheduling (seg -> match -> rigid)

    Uses Adam optimizer with lr=1e-4, cosine annealing lr scheduler,
    gradient clipping, and model checkpointing based on validation loss.

    Input:
        config: configuration object with training parameters
    """
    train_loader, val_loader = build_data_loaders(config)
    model = build_jigsaw_model(config)

    model_save_path = config.MODEL_SAVE_PATH
    results_save_path = config.OUTPUT_PATH

    if config.LOG_FILE_NAME is not None and len(config.LOG_FILE_NAME) > 0:
        logger_name = f"{config.MODEL_NAME}_{config.LOG_FILE_NAME}"
    else:
        logger_name = f"{config.MODEL_NAME}_{NOW_TIME}"
    logger = CSVLogger(
        save_dir=results_save_path,
        name=logger_name,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="model{epoch:03d}",
        monitor=config.CALLBACK.CHECKPOINT_MONITOR,
        save_top_k=10, # save top 10 checkpoints
        mode=config.CALLBACK.CHECKPOINT_MODE,
        save_last=True, # always save last checkpoint
    )
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        checkpoint_callback,
    ]

    training_log_dict = {
        'logger': logger,
        'accelerator': 'gpu',
        'devices': list(config.GPUS),
        'max_epochs': config.TRAIN.NUM_EPOCHS,
        'callbacks': callbacks,
        'benchmark': config.CUDNN,
        'gradient_clip_val': config.TRAIN.CLIP_GRAD,
        'check_val_every_n_epoch': config.TRAIN.VAL_EVERY,
        'log_every_n_steps': 10,
        # 'profiler': 'simple',
        # 'detect_anomaly': True,
        'benchmark': True,  # enable cuDNN benchmark for speed
    }

    if getattr(config, 'FP16', False):
        training_log_dict['precision'] = '16-mixed'

    trainer = pl.Trainer(**training_log_dict)

    ckp_files = os.listdir(model_save_path)
    ckp_files = [
        ckp for ckp in ckp_files if ("model_" in ckp) or ("last" in ckp)
    ]

    if config.WEIGHT_FILE:
        ckp = torch.load(config.WEIGHT_FILE, map_location='cpu', weights_only=False)

        if 'state_dict' in ckp.keys():
            # full checkpoint with optimizer etc.
            ckp_path = config.WEIGHT_FILE
        else:
            # only model weights: start training from scratch with these weights
            ckp_path = None
            result = model.load_state_dict(ckp, strict=False)

            if result.missing_keys:
                print(f"WARNING: {len(result.missing_keys)} missing keys when loading weights")
            if result.unexpected_keys:
                print(f"WARNING: {len(result.unexpected_keys)} unexpected keys when loading weights")
            if not result.missing_keys and not result.unexpected_keys:
                print("INFO: All weights loaded successfully for fine-tuning")

            # clone pretrained attention layers into new double-attention layers if absent from checkpoint
            if hasattr(model, 'tf_self2') and hasattr(model, 'tf_self1'):
                if any('tf_self2' in k for k in result.missing_keys):
                    model.tf_self2.load_state_dict(model.tf_self1.state_dict())
            if hasattr(model, 'tf_cross2') and hasattr(model, 'tf_cross1'):
                if any('tf_cross2' in k for k in result.missing_keys):
                    model.tf_cross2.load_state_dict(model.tf_cross1.state_dict())
    elif ckp_files: # load from last checkpoint in model save path
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
        )
        last_ckp = ckp_files[-1]
        ckp_path = os.path.join(model_save_path, last_ckp)
    else:
        ckp_path = None # start training from scratch

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckp_path)
    print("Done training.")


if __name__ == "__main__":
    args = parse_args("Jigsaw")
    pl.seed_everything(CONFIG.RANDOM_SEED)

    file_end = NOW_TIME
    if CONFIG.LOG_FILE_NAME is not None and len(CONFIG.LOG_FILE_NAME) > 0:
        file_end += "_{}".format(CONFIG.LOG_FILE_NAME)
    log_file = f"train_log_{file_end}"

    with DuplicateStdoutFileManager(os.path.join(CONFIG.OUTPUT_PATH, f"{log_file}.log")) as _:
        print_edict(CONFIG)
        train_model(CONFIG)