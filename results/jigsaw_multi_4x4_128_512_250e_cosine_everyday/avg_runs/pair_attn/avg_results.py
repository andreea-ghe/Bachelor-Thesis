import numpy as np


list_of_results = [
    {
      "test/chamfer_distance": 0.07024271786212921,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.017261100932955742,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.899490356445312,
      "test/mat_f1": 0.09270813316106796,
      "test/mat_loss": 9.882229804992676,
      "test/mat_precision": 0.09270817041397095,
      "test/mat_recall": 0.09270817041397095,
      "test/n_critical_max": 117.26856994628906,
      "test/part_acc": 0.753333330154419,
      "test/rot_mae": 23.156986236572266,
      "test/rot_mse": 2537.5986328125,
      "test/rot_rmse": 26.61665916442871,
      "test/trans_mae": 0.049037374556064606,
      "test/trans_mse": 0.01821325346827507,
      "test/trans_rmse": 0.06635279208421707
    },
    {
      "test/chamfer_distance": 0.07311371713876724,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.017017727717757225,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 10.09384822845459,
      "test/mat_f1": 0.09043881297111511,
      "test/mat_loss": 10.07682991027832,
      "test/mat_precision": 0.0904388576745987,
      "test/mat_recall": 0.0904388576745987,
      "test/n_critical_max": 117.29356384277344,
      "test/part_acc": 0.7509469985961914,
      "test/rot_mae": 23.47216796875,
      "test/rot_mse": 2656.916748046875,
      "test/rot_rmse": 27.22876739501953,
      "test/trans_mae": 0.04887278378009796,
      "test/trans_mse": 0.018804039806127548,
      "test/trans_rmse": 0.06769894063472748
    },
    {
      "test/chamfer_distance": 0.07358713448047638,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.01716235652565956,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.511629104614258,
      "test/mat_f1": 0.09410306066274643,
      "test/mat_loss": 9.494466781616211,
      "test/mat_precision": 0.09410310536623001,
      "test/mat_recall": 0.09410310536623001,
      "test/n_critical_max": 117.32251739501953,
      "test/part_acc": 0.7557252049446106,
      "test/rot_mae": 22.813823699951172,
      "test/rot_mse": 2528.02587890625,
      "test/rot_rmse": 26.484027862548828,
      "test/trans_mae": 0.050274573266506195,
      "test/trans_mse": 0.019266055896878242,
      "test/trans_rmse": 0.06823490560054779
    },
    {
      "test/chamfer_distance": 0.07108492404222488,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.01715647056698799,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 10.359392166137695,
      "test/mat_f1": 0.08628690987825394,
      "test/mat_loss": 10.34223461151123,
      "test/mat_precision": 0.08628695458173752,
      "test/mat_recall": 0.08628695458173752,
      "test/n_critical_max": 117.41108703613281,
      "test/part_acc": 0.7562141418457031,
      "test/rot_mae": 24.464248657226562,
      "test/rot_mse": 2809.3232421875,
      "test/rot_rmse": 28.442440032958984,
      "test/trans_mae": 0.048897624015808105,
      "test/trans_mse": 0.018091365694999695,
      "test/trans_rmse": 0.06637296080589294
    },
    {
      "test/chamfer_distance": 0.06858086585998535,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.017092516645789146,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.097576141357422,
      "test/mat_f1": 0.08465837687253952,
      "test/mat_loss": 9.080484390258789,
      "test/mat_precision": 0.0846584141254425,
      "test/mat_recall": 0.0846584141254425,
      "test/n_critical_max": 120.13497924804688,
      "test/part_acc": 0.7566539645195007,
      "test/rot_mae": 23.31964874267578,
      "test/rot_mse": 2625.798583984375,
      "test/rot_rmse": 27.20726203918457,
      "test/trans_mae": 0.04809015244245529,
      "test/trans_mse": 0.018025841563940048,
      "test/trans_rmse": 0.06581207364797592
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