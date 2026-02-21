import numpy as np

list_of_results = [
    {
      "test/chamfer_distance": 0.09972307831048965,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.031347163021564484,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 8.888677597045898,
      "test/mat_f1": 0.03354622423648834,
      "test/mat_loss": 8.857332229614258,
      "test/mat_precision": 0.033546265214681625,
      "test/mat_recall": 0.033546265214681625,
      "test/n_critical_max": 131.84091186523438,
      "test/part_acc": 0.6344696879386902,
      "test/rot_mae": 32.98208999633789,
      "test/rot_mse": 3695.87158203125,
      "test/rot_rmse": 37.952457427978516,
      "test/trans_mae": 0.07344377040863037,
      "test/trans_mse": 0.026502905413508415,
      "test/trans_rmse": 0.0916389748454094
    },
    {
      "test/chamfer_distance": 0.09768573194742203,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.03107415698468685,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 7.109078884124756,
      "test/mat_f1": 0.039577506482601166,
      "test/mat_loss": 7.078004360198975,
      "test/mat_precision": 0.03957754001021385,
      "test/mat_recall": 0.03957754001021385,
      "test/n_critical_max": 130.46893310546875,
      "test/part_acc": 0.6337099671363831,
      "test/rot_mae": 32.94063949584961,
      "test/rot_mse": 3778.686279296875,
      "test/rot_rmse": 38.14662551879883,
      "test/trans_mae": 0.0725414827466011,
      "test/trans_mse": 0.025403587147593498,
      "test/trans_rmse": 0.09020166844129562
    },
    {
      "test/chamfer_distance": 0.09432332962751389,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.03085339441895485,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 8.338972091674805,
      "test/mat_f1": 0.03488245978951454,
      "test/mat_loss": 8.308117866516113,
      "test/mat_precision": 0.034882497042417526,
      "test/mat_recall": 0.034882497042417526,
      "test/n_critical_max": 127.07344818115234,
      "test/part_acc": 0.6497175097465515,
      "test/rot_mae": 32.379150390625,
      "test/rot_mse": 3650.913818359375,
      "test/rot_rmse": 37.530906677246094,
      "test/trans_mae": 0.07047109305858612,
      "test/trans_mse": 0.024907933548092842,
      "test/trans_rmse": 0.08801908791065216
    },
    {
      "test/chamfer_distance": 0.09720564633607864,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.03160342574119568,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 8.491998672485352,
      "test/mat_f1": 0.03911241516470909,
      "test/mat_loss": 8.460394859313965,
      "test/n_critical_max": 131.5530242919922,
      "test/part_acc": 0.6448863744735718,
      "test/rot_mae": 31.75408363342285,
      "test/rot_mse": 3592.270263671875,
      "test/rot_rmse": 36.85091018676758,
      "test/trans_mae": 0.06947144120931625,
      "test/trans_mse": 0.02473902329802513,
      "test/trans_rmse": 0.08601399511098862
    },
    {
      "test/chamfer_distance": 0.09093771129846573,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.03120429255068302,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.307951927185059,
      "test/mat_f1": 0.03665882349014282,
      "test/mat_loss": 9.276747703552246,
      "test/mat_precision": 0.03665886074304581,
      "test/mat_recall": 0.03665886074304581,
      "test/n_critical_max": 132.78720092773438,
      "test/part_acc": 0.6440678238868713,
      "test/rot_mae": 31.336599349975586,
      "test/rot_mse": 3478.1962890625,
      "test/rot_rmse": 36.5550651550293,
      "test/trans_mae": 0.07037607580423355,
      "test/trans_mse": 0.024345291778445244,
      "test/trans_rmse": 0.08760225772857666
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