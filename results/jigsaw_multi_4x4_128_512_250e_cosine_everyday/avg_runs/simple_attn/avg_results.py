import numpy as np

list_of_results = [
    {
      "test/chamfer_distance": 0.07176706939935684,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.017431654036045074,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 8.744424819946289,
      "test/mat_f1": 0.09362717717885971,
      "test/mat_loss": 8.726993560791016,
      "test/mat_precision": 0.0936272144317627,
      "test/mat_recall": 0.0936272144317627,
      "test/n_critical_max": 119.45769500732422,
      "test/part_acc": 0.7538461685180664,
      "test/rot_mae": 23.288057327270508,
      "test/rot_mse": 2620.6171875,
      "test/rot_rmse": 27.122270584106445,
      "test/trans_mae": 0.04855597764253616,
      "test/trans_mse": 0.01834687404334545,
      "test/trans_rmse": 0.06573035567998886
    },
    {
      "test/chamfer_distance": 0.07720872759819031,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.017309138551354408,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.842533111572266,
      "test/mat_f1": 0.0914081260561943,
      "test/mat_loss": 9.825223922729492,
      "test/mat_precision": 0.0914081558585167,
      "test/mat_recall": 0.0914081558585167,
      "test/n_critical_max": 120.4914321899414,
      "test/part_acc": 0.7523809671401978,
      "test/rot_mae": 23.664493560791016,
      "test/rot_mse": 2656.781494140625,
      "test/rot_rmse": 27.58184814453125,
      "test/trans_mae": 0.049152206629514694,
      "test/trans_mse": 0.0194866843521595,
      "test/trans_rmse": 0.06818824261426926
    },
    {
      "test/chamfer_distance": 0.06580787152051926,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.017466802150011063,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.751976013183594,
      "test/mat_f1": 0.09261883795261383,
      "test/mat_loss": 9.734508514404297,
      "test/mat_precision": 0.09261887520551682,
      "test/mat_recall": 0.09261887520551682,
      "test/n_critical_max": 120.63047790527344,
      "test/part_acc": 0.7590476274490356,
      "test/rot_mae": 23.623424530029297,
      "test/rot_mse": 2646.9619140625,
      "test/rot_rmse": 27.476192474365234,
      "test/trans_mae": 0.0467420369386673,
      "test/trans_mse": 0.017226221039891243,
      "test/trans_rmse": 0.0638870820403099        
    },
    {
      "test/chamfer_distance": 0.07016151398420334,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.017152732238173485,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.13874626159668,
      "test/mat_f1": 0.08631350845098495,
      "test/mat_loss": 9.121594429016113,
      "test/mat_precision": 0.08631355315446854,
      "test/mat_recall": 0.08631355315446854,
      "test/n_critical_max": 119.62213897705078,
      "test/part_acc": 0.7566794157028198,
      "test/rot_mae": 23.660724639892578,
      "test/rot_mse": 2631.9189453125,
      "test/rot_rmse": 27.30084800720215,
      "test/trans_mae": 0.04887920245528221,
      "test/trans_mse": 0.01821354776620865,
      "test/trans_rmse": 0.06630094349384308
    }
]

avg_results = {}
for result in list_of_results:
    for key, value in result.items():
        if key not in avg_results:
            avg_results[key] = []
        avg_results[key].append(value)

new_results = {}
for key, value in avg_results.items():
    new_results[key] = {}
    new_results[key]["avg"] = sum(value) / len(value)
    new_results[key]["std"] = np.std(value)

for key, value in new_results.items():
    print(f"{key}")
    print(f"avg: {value['avg']}")
    print(f"std: {value['std']}")
    print("--------------------------------")