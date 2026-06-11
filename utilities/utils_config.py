import yaml
import importlib
from easydict import EasyDict as edict
from ast import literal_eval


__C = edict()
CONFIG = __C # global config object

__C.MODEL_NAME = ""  # this name would be the result file name
__C.MODULE = ""  # sample: dgl.network  b_global.network

__C.BATCH_SIZE = 32
__C.NUM_WORKERS = 8

__C.LOG_FILE_NAME = ""  # the suffix of log file

__C.MODEL_SAVE_PATH = ""  # auto generated

#
# Dataset
#
__C.DATASET = ""

# wandb project name
__C.PROJECT = ""

#
# training options
#
__C.TRAIN = edict()

__C.TRAIN.NUM_EPOCHS = 200          # total epochs
__C.TRAIN.OPTIMIZER = "SGD"         # optimizer type
__C.TRAIN.LR = 0.001               # start learning rate
__C.TRAIN.PAIR_LR_MULT = 10.0      # LR multiplier for pair_geometric_encoder (trains from scratch)
__C.TRAIN.LR_SCHEDULER = "cosine"   # LR scheduler
__C.TRAIN.LR_DECAY = 100.0         # learning rate decay
__C.TRAIN.LR_STEP = [10, 20]       # learning rate decay step (in epochs)
__C.TRAIN.WARMUP_RATIO = 0.0       # warmup ratio for Adam Cosine
__C.TRAIN.CLIP_GRAD = None         # gradient clipping
__C.TRAIN.beta1 = 0                # beta1 for Adam optimizer
__C.TRAIN.beta2 = 0.9              # beta2 for Adam optimizer
__C.TRAIN.WEIGHT_DECAY = 0.0       # weight decay for Adam or SGD
__C.TRAIN.MOMENTUM = 0.9           # SGD momentum
__C.TRAIN.VAL_EVERY = 5            # check val every n epoch
__C.TRAIN.VIS = True                # visualization during training
__C.TRAIN.VAL_SAMPLE_VIS = 5
__C.TRAIN.LOSS = ""                 # loss function

#
# callback
#
__C.CALLBACK = edict()
__C.CALLBACK.MATCHING_TASK = ["trans"]
__C.CALLBACK.CHECKPOINT_MONITOR = "val/loss"
__C.CALLBACK.CHECKPOINT_MODE = "min"

#
# loss config
#
__C.LOSS = edict()

#
# evaluation options
#
__C.EVAL = edict()

#
# misc
#
__C.GPUS = [0]                      # parallel GPU indices ([0] for single GPU)
__C.PARALLEL_STRATEGY = "ddp"       # parallel strategy for multiple GPUs
__C.FP16 = False                    # float precision: 32 for False, 16 for True
__C.CUDNN = False                   # cuDNN benchmark

__C.WEIGHT_FILE = ""
__C.OUTPUT_PATH = ""                # output path (for checkpoints, running logs)
__C.STATISTIC_STEP = 100            # iteration step to print running statistics
__C.RANDOM_SEED = 42                # random seed
__C.STATS = ""                      # directory for collecting statistics of results


def merge_configs(src, dest):
    """Merge source config into destination config recursively."""
    for key, value in src.items():
        if key not in dest:
            raise KeyError(f'Key {key} not a valid config key.')

        if not isinstance(value, type(dest[key])) and not (isinstance(value, dict) and isinstance(dest[key], dict)):
            if isinstance(dest[key], float) and isinstance(value, int):
                value = float(value)
            else:
                if key not in ['CLASS']:
                    raise ValueError(f'Type mismatch ({type(dest[key])} vs. {type(value)}) for config key: {key}')
        
        if isinstance(value, dict):
            try:
                merge_configs(src[key], dest[key])
            except:
                print(f'Error under config key: {key}')
                raise
        else:
            dest[key] = value


def config_from_file(filename):
    """Load configuration from a YAML file and merge into the default config."""
    with open(filename, 'r') as f:
        config = edict(yaml.full_load(f))

    # dynamically import model and dataset configs
    if 'MODULE' in config and 'MODEL' not in __C:
        # model_config_module = '.'.join(['model'] + [config.MODULE.split('.')[0]] + ['model_config'])
        # model_config_module = '.'.join([config.MODULE.split('.')[0]] + ['model_config'])
        model_config_module = '.'.join([config.MODULE, 'model_config'])
        mod = importlib.import_module(model_config_module)
        __C['MODEL'] = mod.get_model_config()

    if 'DATASET' in config and config.DATASET is not None:
        mod = importlib.import_module('dataset_preprocessing')
        # __C['DATA'] = mod.dataset_config[config.DATASET.split('.')[0].upper()]
        __C['DATA'] = getattr(mod.dataset_config.dataset_cfg, config.DATASET.split('.')[0].upper())


    merge_configs(config, __C)

def config_from_list(config_list):
    """Override config values from a flat key-value list (e.g. ['TRAIN.LR', '0.01'])."""
    assert len(config_list) % 2 == 0
    
    for key, v in zip(config_list[0::2], config_list[1::2]):
        key_list = key.split('.')
        d = __C

        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]

        subkey = key_list[-1]
        assert subkey in d.keys()

        try:
            value = literal_eval(v)
        except:
            value = v

        assert type(value) == type(d[subkey]), f'Type mismatch ({type(d[subkey])} vs. {type(value)}) for config key: {key}'
        d[subkey] = value