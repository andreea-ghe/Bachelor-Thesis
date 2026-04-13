Loading: results/jigsaw_finetune_everyday_double_attn_correct_ds/model_save/modelepoch=099.ckpt

============================================================
GATE VALUES (0 = layer bypassed, >0 = layer active)
============================================================
  gate_self2: 0.061662
  gate_cross2: -0.217126

============================================================
WEIGHT DIVERGENCE (tf_self2 vs tf_self1, tf_cross2 vs tf_cross1)
============================================================
  tf_self2 vs tf_self1:
    Absolute L2 diff (sum over 29 params): 6938.6599
    Relative divergence: 0.31%
    --> Layers barely changed from initialization (< 1%)
  tf_cross2 vs tf_cross1:
    Absolute L2 diff (sum over 12 params): 281.8813
    Relative divergence: 76.20%
    --> Layers significantly diverged from copies

============================================================
PAIR GEOMETRIC ENCODER
============================================================
  NOT FOUND (no pair attention in this checkpoint)

============================================================
QUICK SUMMARY
============================================================
  Gates are open (0.0617, -0.2171): layers are actively used.