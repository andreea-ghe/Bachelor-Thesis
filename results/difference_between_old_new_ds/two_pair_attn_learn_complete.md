Found 11 checkpoints in results/jigsaw_finetune_everyday_pair_attn_correct_ds/model_save

================================================================================
[modelepoch=054.ckpt]  (epoch=54)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
  bias_proj.0.bias:
    mean=-0.024476  std=0.112018  min=-0.270271  max=+0.203311  norm=0.638871
  bias_proj.0.weight:
    mean=-0.020101  std=0.210710  min=-1.482329  max=+1.641483  norm=6.770045
  bias_proj.2.bias:
    mean=+0.057744  std=0.000000  min=+0.057744  max=+0.057744  norm=0.057744
  bias_proj.2.weight:
    mean=+0.013175  std=0.282954  min=-0.442518  max=+0.552924  norm=1.577185
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 4.275649398050857e-05
    pg0_initial_lr: 0.0001
    pg1_lr: 0.0004275649398050857
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.448625  std=0.536803  range=[+0.106590, +1.487891]  abs_mean=0.448625
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=059.ckpt]  (epoch=59)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.024195  std=0.117934  min=-0.270271  max=+0.232903  norm=0.670741
    Δ from first ckpt:  mean_abs_diff=0.00491316  max_diff=0.05438349
  bias_proj.0.weight:
    mean=-0.020419  std=0.214744  min=-1.507914  max=+1.709444  norm=6.899470
    Δ from first ckpt:  mean_abs_diff=0.00290521  max_diff=0.09135589
  bias_proj.2.bias:
    mean=+0.100517  std=0.000000  min=+0.100517  max=+0.100517  norm=0.100517
    Δ from first ckpt:  mean_abs_diff=0.04277220  max_diff=0.04277220
  bias_proj.2.weight:
    mean=+0.007380  std=0.291421  min=-0.497747  max=+0.552924  norm=1.623098
    Δ from first ckpt:  mean_abs_diff=0.00931205  max_diff=0.05953960
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 3.520365877844012e-05
    pg0_initial_lr: 0.0001
    pg1_lr: 0.00035203658778440114
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.257581  std=0.077321  range=[+0.122821, +0.333007]  abs_mean=0.257581
    ⚠️  LOW VARIANCE — bias is nearly constant, minimal effect on attention

================================================================================
[modelepoch=064.ckpt]  (epoch=64)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.025721  std=0.118349  min=-0.270271  max=+0.233360  norm=0.674810
    Δ from first ckpt:  mean_abs_diff=0.00555881  max_diff=0.06986362
  bias_proj.0.weight:
    mean=-0.020536  std=0.217242  min=-1.525550  max=+1.745201  norm=6.979371
    Δ from first ckpt:  mean_abs_diff=0.00305511  max_diff=0.11496821
  bias_proj.2.bias:
    mean=+0.106123  std=0.000000  min=+0.106123  max=+0.106123  norm=0.106123
    Δ from first ckpt:  mean_abs_diff=0.04837902  max_diff=0.04837902
  bias_proj.2.weight:
    mean=+0.011801  std=0.293059  min=-0.507226  max=+0.552924  norm=1.633049
    Δ from first ckpt:  mean_abs_diff=0.01057846  max_diff=0.06610487
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 2.8027470262892437e-05
    pg0_initial_lr: 0.0001
    pg1_lr: 0.0002802747026289244
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.374402  std=0.076840  range=[+0.252873, +0.451655]  abs_mean=0.374402
    ⚠️  LOW VARIANCE — bias is nearly constant, minimal effect on attention

================================================================================
[modelepoch=069.ckpt]  (epoch=69)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.023834  std=0.120126  min=-0.270271  max=+0.245809  norm=0.682290
    Δ from first ckpt:  mean_abs_diff=0.00796404  max_diff=0.06986362
  bias_proj.0.weight:
    mean=-0.020604  std=0.218829  min=-1.543702  max=+1.763933  norm=7.030083
    Δ from first ckpt:  mean_abs_diff=0.00383873  max_diff=0.12417912
  bias_proj.2.bias:
    mean=+0.119926  std=0.000000  min=+0.119926  max=+0.119926  norm=0.119926
    Δ from first ckpt:  mean_abs_diff=0.06218193  max_diff=0.06218193
  bias_proj.2.weight:
    mean=+0.014598  std=0.293925  min=-0.511515  max=+0.550725  norm=1.638586
    Δ from first ckpt:  mean_abs_diff=0.01746681  max_diff=0.10429798
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 2.1404630011522586e-05
    pg0_initial_lr: 0.0001
    pg1_lr: 0.00021404630011522585
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.453517  std=0.098092  range=[+0.330263, +0.604024]  abs_mean=0.453517
    ⚠️  LOW VARIANCE — bias is nearly constant, minimal effect on attention

================================================================================
[modelepoch=074.ckpt]  (epoch=74)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.021027  std=0.122957  min=-0.270271  max=+0.267330  norm=0.694853
    Δ from first ckpt:  mean_abs_diff=0.01032457  max_diff=0.06986362
  bias_proj.0.weight:
    mean=-0.020469  std=0.220237  min=-1.560384  max=+1.780284  norm=7.074525
    Δ from first ckpt:  mean_abs_diff=0.00433885  max_diff=0.14086139
  bias_proj.2.bias:
    mean=+0.147021  std=0.000000  min=+0.147021  max=+0.147021  norm=0.147021
    Δ from first ckpt:  mean_abs_diff=0.08927631  max_diff=0.08927631
  bias_proj.2.weight:
    mean=+0.016731  std=0.296607  min=-0.516752  max=+0.550725  norm=1.654148
    Δ from first ckpt:  mean_abs_diff=0.02159348  max_diff=0.14617515
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 1.54982143312659e-05
    pg0_initial_lr: 0.0001
    pg1_lr: 0.000154982143312659
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.550020  std=0.241235  range=[+0.284026, +0.981623]  abs_mean=0.550020
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=079.ckpt]  (epoch=79)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.020861  std=0.123709  min=-0.270271  max=+0.274805  norm=0.698821
    Δ from first ckpt:  mean_abs_diff=0.01127395  max_diff=0.07149418
  bias_proj.0.weight:
    mean=-0.020578  std=0.220656  min=-1.568215  max=+1.787676  norm=7.088184
    Δ from first ckpt:  mean_abs_diff=0.00450537  max_diff=0.14869177
  bias_proj.2.bias:
    mean=+0.154193  std=0.000000  min=+0.154193  max=+0.154193  norm=0.154193
    Δ from first ckpt:  mean_abs_diff=0.09644829  max_diff=0.09644829
  bias_proj.2.weight:
    mean=+0.017612  std=0.295484  min=-0.508608  max=+0.550725  norm=1.648199
    Δ from first ckpt:  mean_abs_diff=0.02093682  max_diff=0.15600507
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 1.0453658778440109e-05
    pg0_initial_lr: 0.0001
    pg1_lr: 0.00010453658778440107
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.519287  std=0.174484  range=[+0.303764, +0.755434]  abs_mean=0.519287
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=084.ckpt]  (epoch=84)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.019873  std=0.123821  min=-0.270271  max=+0.274517  norm=0.698509
    Δ from first ckpt:  mean_abs_diff=0.01133228  max_diff=0.07120685
  bias_proj.0.weight:
    mean=-0.020459  std=0.222191  min=-1.586760  max=+1.806142  norm=7.136749
    Δ from first ckpt:  mean_abs_diff=0.00499521  max_diff=0.16723669
  bias_proj.2.bias:
    mean=+0.153948  std=0.000000  min=+0.153948  max=+0.153948  norm=0.153948
    Δ from first ckpt:  mean_abs_diff=0.09620379  max_diff=0.09620379
  bias_proj.2.weight:
    mean=+0.019437  std=0.300670  min=-0.521510  max=+0.550725  norm=1.677668
    Δ from first ckpt:  mean_abs_diff=0.02777118  max_diff=0.20758417
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 6.395177052675795e-06
    pg0_initial_lr: 0.0001
    pg1_lr: 6.395177052675794e-05
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.594534  std=0.116798  range=[+0.462027, +0.780679]  abs_mean=0.594534
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=089.ckpt]  (epoch=89)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.020600  std=0.123394  min=-0.270271  max=+0.273441  norm=0.696839
    Δ from first ckpt:  mean_abs_diff=0.01112116  max_diff=0.07013021
  bias_proj.0.weight:
    mean=-0.020530  std=0.222428  min=-1.591387  max=+1.810419  norm=7.144479
    Δ from first ckpt:  mean_abs_diff=0.00511435  max_diff=0.17186439
  bias_proj.2.bias:
    mean=+0.151605  std=0.000000  min=+0.151605  max=+0.151605  norm=0.151605
    Δ from first ckpt:  mean_abs_diff=0.09386072  max_diff=0.09386072
  bias_proj.2.weight:
    mean=+0.019585  std=0.300533  min=-0.520195  max=+0.550725  norm=1.676962
    Δ from first ckpt:  mean_abs_diff=0.02768599  max_diff=0.20782015
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 3.4227024433899e-06
    pg0_initial_lr: 0.0001
    pg1_lr: 3.4227024433899005e-05
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.484793  std=0.116939  range=[+0.316307, +0.612301]  abs_mean=0.484793
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=094.ckpt]  (epoch=94)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.020720  std=0.123417  min=-0.270271  max=+0.272964  norm=0.697081
    Δ from first ckpt:  mean_abs_diff=0.01110412  max_diff=0.06986362
  bias_proj.0.weight:
    mean=-0.020588  std=0.222548  min=-1.592467  max=+1.811426  norm=7.148468
    Δ from first ckpt:  mean_abs_diff=0.00514088  max_diff=0.17294431
  bias_proj.2.bias:
    mean=+0.151765  std=0.000000  min=+0.151765  max=+0.151765  norm=0.151765
    Δ from first ckpt:  mean_abs_diff=0.09402046  max_diff=0.09402046
  bias_proj.2.weight:
    mean=+0.019888  std=0.300704  min=-0.519094  max=+0.550725  norm=1.678023
    Δ from first ckpt:  mean_abs_diff=0.02804011  max_diff=0.21562207
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 1.6094271405406859e-06
    pg0_initial_lr: 0.0001
    pg1_lr: 1.609427140540686e-05
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.474791  std=0.121887  range=[+0.327523, +0.618887]  abs_mean=0.474791
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=099.ckpt]  (epoch=99)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.020941  std=0.123264  min=-0.270271  max=+0.271973  norm=0.696454
    Δ from first ckpt:  mean_abs_diff=0.01098787  max_diff=0.06986362
  bias_proj.0.weight:
    mean=-0.020617  std=0.222678  min=-1.594329  max=+1.813227  norm=7.152699
    Δ from first ckpt:  mean_abs_diff=0.00516673  max_diff=0.17480624
  bias_proj.2.bias:
    mean=+0.150735  std=0.000000  min=+0.150735  max=+0.150735  norm=0.150735
    Δ from first ckpt:  mean_abs_diff=0.09299050  max_diff=0.09299050
  bias_proj.2.weight:
    mean=+0.019963  std=0.301006  min=-0.519626  max=+0.550725  norm=1.679734
    Δ from first ckpt:  mean_abs_diff=0.02841581  max_diff=0.21901992
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 0.0001
    pg0_initial_lr: 0.0001
    pg1_lr: 0.001
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.662434  std=0.316140  range=[+0.328481, +1.172724]  abs_mean=0.662434
    ✅  Good variance — bias differentiates between pairs

================================================================================
[last.ckpt]  (last)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.020941  std=0.123264  min=-0.270271  max=+0.271973  norm=0.696454
    Δ from first ckpt:  mean_abs_diff=0.01098787  max_diff=0.06986362
  bias_proj.0.weight:
    mean=-0.020617  std=0.222678  min=-1.594329  max=+1.813227  norm=7.152699
    Δ from first ckpt:  mean_abs_diff=0.00516673  max_diff=0.17480624
  bias_proj.2.bias:
    mean=+0.150735  std=0.000000  min=+0.150735  max=+0.150735  norm=0.150735
    Δ from first ckpt:  mean_abs_diff=0.09299050  max_diff=0.09299050
  bias_proj.2.weight:
    mean=+0.019963  std=0.301006  min=-0.519626  max=+0.550725  norm=1.679734
    Δ from first ckpt:  mean_abs_diff=0.02841581  max_diff=0.21901992
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 0.0001
    pg0_initial_lr: 0.0001
    pg1_lr: 0.001
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.476846  std=0.139645  range=[+0.216903, +0.623645]  abs_mean=0.476846
    ✅  Good variance — bias differentiates between pairs


================================================================================
SUMMARY
================================================================================

Weight changes (first → last checkpoint):
  angle_rbf.centers: mean_abs_diff=0.00000000  → ❌ FROZEN — not learning at all
  bias_proj.0.bias: mean_abs_diff=0.01098787  → ✅ LEARNING — significant weight changes
  bias_proj.0.weight: mean_abs_diff=0.00516673  → 🟡 LEARNING SLOWLY — some movement
  bias_proj.2.bias: mean_abs_diff=0.09299050  → ✅ LEARNING — significant weight changes
  bias_proj.2.weight: mean_abs_diff=0.02841581  → ✅ LEARNING — significant weight changes
  distance_rbf.centers: mean_abs_diff=0.00000000  → ❌ FROZEN — not learning at all