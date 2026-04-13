Found 11 checkpoints in results/jigsaw_finetune_multi_everyday_pair_attn_correct_ds/model_save

================================================================================
[modelepoch=054.ckpt]  (epoch=54)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
  bias_proj.0.bias:
    mean=-0.047489  std=0.145053  min=-0.306346  max=+0.458081  norm=0.851126
  bias_proj.0.weight:
    mean=-0.032595  std=0.251711  min=-2.003844  max=+2.067752  norm=8.118093
  bias_proj.2.bias:
    mean=+0.105878  std=0.000000  min=+0.105878  max=+0.105878  norm=0.105878
  bias_proj.2.weight:
    mean=+0.046884  std=0.324522  min=-0.456964  max=+0.603353  norm=1.826225
  distance_rbf.centers:
    mean=+5.000000  std=3.173968  min=+0.000000  max=+10.000000  norm=23.475756

  LR info:
    lr_ratios: [1.0, 10.0]
    pg0_lr: 4.275649398050857e-05
    pg0_initial_lr: 0.0001
    pg1_lr: 0.0004275649398050857
    pg1_initial_lr: 0.001

  Output bias (dummy input):
    mean=+0.735128  std=0.149995  range=[+0.485716, +0.865849]  abs_mean=0.735128
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=059.ckpt]  (epoch=59)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.048630  std=0.142219  min=-0.306346  max=+0.446548  norm=0.838269
    Δ from first ckpt:  mean_abs_diff=0.00263970  max_diff=0.02460521
  bias_proj.0.weight:
    mean=-0.032668  std=0.254561  min=-2.019031  max=+2.131258  norm=8.208819
    Δ from first ckpt:  mean_abs_diff=0.00102211  max_diff=0.06623518
  bias_proj.2.bias:
    mean=+0.118720  std=0.000000  min=+0.118720  max=+0.118720  norm=0.118720
    Δ from first ckpt:  mean_abs_diff=0.01284191  max_diff=0.01284191
  bias_proj.2.weight:
    mean=+0.044829  std=0.325120  min=-0.501325  max=+0.596011  norm=1.827867
    Δ from first ckpt:  mean_abs_diff=0.00353373  max_diff=0.04436037
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
    mean=+0.708412  std=0.053819  range=[+0.595287, +0.746647]  abs_mean=0.708412
    ⚠️  LOW VARIANCE — bias is nearly constant, minimal effect on attention

================================================================================
[modelepoch=064.ckpt]  (epoch=64)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.048959  std=0.142333  min=-0.306346  max=+0.450943  norm=0.839478
    Δ from first ckpt:  mean_abs_diff=0.00282920  max_diff=0.03686538
  bias_proj.0.weight:
    mean=-0.032253  std=0.257241  min=-2.019240  max=+2.198285  norm=8.292169
    Δ from first ckpt:  mean_abs_diff=0.00158720  max_diff=0.13430047
  bias_proj.2.bias:
    mean=+0.124122  std=0.000000  min=+0.124122  max=+0.124122  norm=0.124122
    Δ from first ckpt:  mean_abs_diff=0.01824427  max_diff=0.01824427
  bias_proj.2.weight:
    mean=+0.046367  std=0.327316  min=-0.510171  max=+0.622822  norm=1.841199
    Δ from first ckpt:  mean_abs_diff=0.00324430  max_diff=0.05320612
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
    mean=+1.189268  std=0.887253  range=[+0.717671, +3.226670]  abs_mean=1.189268
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=069.ckpt]  (epoch=69)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049736  std=0.139609  min=-0.306346  max=+0.428385  norm=0.826660
    Δ from first ckpt:  mean_abs_diff=0.00401392  max_diff=0.03934960
  bias_proj.0.weight:
    mean=-0.032291  std=0.258954  min=-2.022749  max=+2.245404  norm=8.346696
    Δ from first ckpt:  mean_abs_diff=0.00184453  max_diff=0.18261659
  bias_proj.2.bias:
    mean=+0.086614  std=0.000000  min=+0.086614  max=+0.086614  norm=0.086614
    Δ from first ckpt:  mean_abs_diff=0.01926418  max_diff=0.01926418
  bias_proj.2.weight:
    mean=+0.044906  std=0.328832  min=-0.550047  max=+0.615081  norm=1.848395
    Δ from first ckpt:  mean_abs_diff=0.00461806  max_diff=0.09308288
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
    mean=+0.546442  std=0.276188  range=[+0.183103, +0.767643]  abs_mean=0.546442
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=074.ckpt]  (epoch=74)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049972  std=0.140232  min=-0.306346  max=+0.433037  norm=0.830375
    Δ from first ckpt:  mean_abs_diff=0.00398814  max_diff=0.04135728
  bias_proj.0.weight:
    mean=-0.032391  std=0.260216  min=-2.022625  max=+2.269521  norm=8.387154
    Δ from first ckpt:  mean_abs_diff=0.00207125  max_diff=0.21393514
  bias_proj.2.bias:
    mean=+0.072764  std=0.000000  min=+0.072764  max=+0.072764  norm=0.072764
    Δ from first ckpt:  mean_abs_diff=0.03311459  max_diff=0.03311459
  bias_proj.2.weight:
    mean=+0.043170  std=0.331556  min=-0.600694  max=+0.611923  norm=1.862108
    Δ from first ckpt:  mean_abs_diff=0.00617912  max_diff=0.14372936
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
    mean=+0.766902  std=0.059508  range=[+0.713225, +0.866793]  abs_mean=0.766902
    ⚠️  LOW VARIANCE — bias is nearly constant, minimal effect on attention

================================================================================
[modelepoch=079.ckpt]  (epoch=79)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049929  std=0.143050  min=-0.306346  max=+0.445835  norm=0.845069
    Δ from first ckpt:  mean_abs_diff=0.00289470  max_diff=0.03204647
  bias_proj.0.weight:
    mean=-0.032485  std=0.260145  min=-2.022625  max=+2.261920  norm=8.385249
    Δ from first ckpt:  mean_abs_diff=0.00221461  max_diff=0.21088493
  bias_proj.2.bias:
    mean=+0.103616  std=0.000000  min=+0.103616  max=+0.103616  norm=0.103616
    Δ from first ckpt:  mean_abs_diff=0.00226207  max_diff=0.00226207
  bias_proj.2.weight:
    mean=+0.041287  std=0.331636  min=-0.624860  max=+0.588716  norm=1.861181
    Δ from first ckpt:  mean_abs_diff=0.00801769  max_diff=0.16789511
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
    mean=+1.181071  std=0.812595  range=[+0.759026, +2.757330]  abs_mean=1.181071
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=084.ckpt]  (epoch=84)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049740  std=0.142752  min=-0.306346  max=+0.443547  norm=0.843143
    Δ from first ckpt:  mean_abs_diff=0.00400024  max_diff=0.03759040
  bias_proj.0.weight:
    mean=-0.032391  std=0.260883  min=-2.023117  max=+2.279359  norm=8.408294
    Δ from first ckpt:  mean_abs_diff=0.00262788  max_diff=0.23043466
  bias_proj.2.bias:
    mean=+0.087023  std=0.000000  min=+0.087023  max=+0.087023  norm=0.087023
    Δ from first ckpt:  mean_abs_diff=0.01885516  max_diff=0.01885516
  bias_proj.2.weight:
    mean=+0.042337  std=0.332797  min=-0.632113  max=+0.596130  norm=1.868349
    Δ from first ckpt:  mean_abs_diff=0.00849169  max_diff=0.17514816
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
    mean=+0.660943  std=0.120615  range=[+0.511013, +0.771469]  abs_mean=0.660943
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=089.ckpt]  (epoch=89)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049720  std=0.143227  min=-0.306346  max=+0.444655  norm=0.845602
    Δ from first ckpt:  mean_abs_diff=0.00357318  max_diff=0.03473425
  bias_proj.0.weight:
    mean=-0.032400  std=0.261050  min=-2.023103  max=+2.280786  norm=8.413644
    Δ from first ckpt:  mean_abs_diff=0.00268788  max_diff=0.23436785
  bias_proj.2.bias:
    mean=+0.088423  std=0.000000  min=+0.088423  max=+0.088423  norm=0.088423
    Δ from first ckpt:  mean_abs_diff=0.01745519  max_diff=0.01745519
  bias_proj.2.weight:
    mean=+0.041657  std=0.332953  min=-0.640723  max=+0.589047  norm=1.868719
    Δ from first ckpt:  mean_abs_diff=0.00914897  max_diff=0.18375865
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
    mean=+0.729416  std=0.116922  range=[+0.513297, +0.869775]  abs_mean=0.729416
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=094.ckpt]  (epoch=94)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049732  std=0.143445  min=-0.306346  max=+0.445451  norm=0.846768
    Δ from first ckpt:  mean_abs_diff=0.00352593  max_diff=0.03473425
  bias_proj.0.weight:
    mean=-0.032442  std=0.261188  min=-2.023103  max=+2.283866  norm=8.418195
    Δ from first ckpt:  mean_abs_diff=0.00269916  max_diff=0.23791862
  bias_proj.2.bias:
    mean=+0.089638  std=0.000000  min=+0.089638  max=+0.089638  norm=0.089638
    Δ from first ckpt:  mean_abs_diff=0.01624058  max_diff=0.01624058
  bias_proj.2.weight:
    mean=+0.041510  std=0.333034  min=-0.643008  max=+0.587906  norm=1.869061
    Δ from first ckpt:  mean_abs_diff=0.00929568  max_diff=0.18604341
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
    mean=+0.884830  std=0.261888  range=[+0.660910, +1.333817]  abs_mean=0.884830
    ✅  Good variance — bias differentiates between pairs

================================================================================
[modelepoch=099.ckpt]  (epoch=99)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049698  std=0.143337  min=-0.306346  max=+0.445727  norm=0.846137
    Δ from first ckpt:  mean_abs_diff=0.00356677  max_diff=0.03512159
  bias_proj.0.weight:
    mean=-0.032436  std=0.261297  min=-2.023121  max=+2.286945  norm=8.421620
    Δ from first ckpt:  mean_abs_diff=0.00271543  max_diff=0.24073315
  bias_proj.2.bias:
    mean=+0.090165  std=0.000000  min=+0.090165  max=+0.090165  norm=0.090165
    Δ from first ckpt:  mean_abs_diff=0.01571312  max_diff=0.01571312
  bias_proj.2.weight:
    mean=+0.041482  std=0.333253  min=-0.646127  max=+0.588184  norm=1.870251
    Δ from first ckpt:  mean_abs_diff=0.00934874  max_diff=0.18916240
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
    mean=+0.719256  std=0.041198  range=[+0.667609, +0.757439]  abs_mean=0.719256
    ⚠️  LOW VARIANCE — bias is nearly constant, minimal effect on attention

================================================================================
[last.ckpt]  (last)
================================================================================
  angle_rbf.centers:
    mean=-0.000000  std=0.634794  min=-1.000000  max=+1.000000  norm=2.458545
    Δ from first ckpt:  mean_abs_diff=0.00000000  max_diff=0.00000000
  bias_proj.0.bias:
    mean=-0.049698  std=0.143337  min=-0.306346  max=+0.445727  norm=0.846137
    Δ from first ckpt:  mean_abs_diff=0.00356677  max_diff=0.03512159
  bias_proj.0.weight:
    mean=-0.032436  std=0.261297  min=-2.023121  max=+2.286945  norm=8.421620
    Δ from first ckpt:  mean_abs_diff=0.00271543  max_diff=0.24073315
  bias_proj.2.bias:
    mean=+0.090165  std=0.000000  min=+0.090165  max=+0.090165  norm=0.090165
    Δ from first ckpt:  mean_abs_diff=0.01571312  max_diff=0.01571312
  bias_proj.2.weight:
    mean=+0.041482  std=0.333253  min=-0.646127  max=+0.588184  norm=1.870251
    Δ from first ckpt:  mean_abs_diff=0.00934874  max_diff=0.18916240
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
    mean=+0.760403  std=0.008229  range=[+0.750195, +0.772969]  abs_mean=0.760403
    ⚠️  LOW VARIANCE — bias is nearly constant, minimal effect on attention


================================================================================
SUMMARY
================================================================================

Weight changes (first → last checkpoint):
  angle_rbf.centers: mean_abs_diff=0.00000000  → ❌ FROZEN — not learning at all
  bias_proj.0.bias: mean_abs_diff=0.00356677  → 🟡 LEARNING SLOWLY — some movement
  bias_proj.0.weight: mean_abs_diff=0.00271543  → 🟡 LEARNING SLOWLY — some movement
  bias_proj.2.bias: mean_abs_diff=0.01571312  → ✅ LEARNING — significant weight changes
  bias_proj.2.weight: mean_abs_diff=0.00934874  → 🟡 LEARNING SLOWLY — some movement
  distance_rbf.centers: mean_abs_diff=0.00000000  → ❌ FROZEN — not learning at al