Loading: results/jigsaw_finetune_everyday_double_attn/model_save/modelepoch=089.ckpt

============================================================
GATE VALUES (0 = layer bypassed, >0 = layer active)
============================================================
  gate_self2: -0.000607
  gate_cross2: -0.011620

============================================================
WEIGHT DIVERGENCE (tf_self2 vs tf_self1, tf_cross2 vs tf_cross1)
============================================================
  tf_self2 vs tf_self1:
    Absolute L2 diff (sum over 29 params): 9256.6569
    Relative divergence: 0.51%
    --> Layers barely changed from initialization (< 1%)
  tf_cross2 vs tf_cross1:
    Absolute L2 diff (sum over 12 params): 180.1664
    Relative divergence: 48.70%
    --> Layers significantly diverged from copies

============================================================
PAIR GEOMETRIC ENCODER
============================================================
  NOT FOUND (no pair attention in this checkpoint)

============================================================
QUICK SUMMARY
============================================================
  Gates are small (-0.0006, -0.0116): layers contribute weakly