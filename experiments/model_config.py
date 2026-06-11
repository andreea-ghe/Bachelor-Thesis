from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

__C.JIGSAW = edict()

# rotation representation type
__C.JIGSAW.ROT_TYPE = 'rmat'

# point-level feature dimension (d = 128)
__C.JIGSAW.PC_FEAT_DIM = 128

# affinity feature dimension after primal-dual descriptor
__C.JIGSAW.AFF_FEAT_DIM = 512

# affinity computation method: 'aff_dual' uses the primal-dual descriptor
__C.JIGSAW.AFFINITY = 'aff_dual'

# PointNet++ with multi-scale grouping; dynamic version handles variable point counts
__C.JIGSAW.ENCODER = 'pointnet2_pt.msg.dynamic'

# whether to use segmentation mask during testing (only fracture points for matching)
__C.JIGSAW.TEST_S_MASK = True

# point classification for fracture surface segmentation
# 'binary': 2-class (fracture vs non-fracture), 'multi': multi-class
__C.JIGSAW.PC_CLS_METHOD = 'binary'
__C.JIGSAW.PC_NUM_CLS = 2

# Sinkhorn algorithm parameters for soft matching
__C.JIGSAW.SINKHORN_MAXITER = 20
__C.JIGSAW.SINKHORN_TAU = 0.05  # temperature parameter

# transformer attention layer parameters
__C.JIGSAW.TF_NUM_HEADS = 8
__C.JIGSAW.TF_NUM_SAMPLE = 16  # neighbor samples for local feature aggregation

# architecture flags (must match checkpoint)
__C.JIGSAW.USE_PAIR_BIAS = False    # geometric bias in cross-attention (pair attention)
__C.JIGSAW.USE_DOUBLE_ATTN = False  # double attention layers (self1->cross1->self2->cross2)
__C.JIGSAW.USE_GABRIEL = False      # Gabriel graph filtering on kNN neighborhoods
__C.JIGSAW.GABRIEL_MIN_KEEP_RATIO = 0.75

# loss weights and scheduling: L = w_cls*L_seg + w_mat*L_mat + w_rig*L_rig
__C.JIGSAW.LOSS = edict()
__C.JIGSAW.LOSS.w_cls_loss = 1.0    # segmentation loss weight
__C.JIGSAW.LOSS.w_mat_loss = 0.0    # matching loss weight (initial, increases after mat_epoch)
__C.JIGSAW.LOSS.mat_epoch = 9       # epoch to activate matching loss
__C.JIGSAW.LOSS.w_rig_loss = 0.0    # rigidity loss weight (initial, increases after rig_epoch)
__C.JIGSAW.LOSS.rig_epoch = 199     # epoch to activate rigidity loss

def get_model_config():
    """
    Get the Jigsaw model configuration.
    
    Returns:
        EasyDict containing all model hyperparameters
    """
    return model_cfg.JIGSAW
