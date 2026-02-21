import numpy as np

list_of_results = [
  {
      "test/chamfer_distance": 0.0935218334197998,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.030991612002253532,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 7.574451923370361,
      "test/mat_f1": 0.03872223198413849,
      "test/mat_loss": 7.543460369110107,
      "test/mat_precision": 0.03872226923704147,
      "test/mat_recall": 0.03872226923704147,
      "test/n_critical_max": 131.39962768554688,
      "test/part_acc": 0.65625,
      "test/rot_mae": 31.434247970581055,
      "test/rot_mse": 3515.1455078125,
      "test/rot_rmse": 36.43903732299805,
      "test/trans_mae": 0.06830879300832748,
      "test/trans_mse": 0.02412777580320835,
      "test/trans_rmse": 0.08537986129522324
  },
  {
      "test/chamfer_distance": 0.09359096735715866,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.031090060248970985,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 8.98166561126709,
      "test/mat_f1": 0.03854789584875107,
      "test/mat_loss": 8.95057487487793,
      "test/mat_precision": 0.03854793310165405,
      "test/mat_recall": 0.03854793310165405,
      "test/n_critical_max": 132.07765197753906,
      "test/part_acc": 0.6401515007019043,
      "test/rot_mae": 31.184045791625977,
      "test/rot_mse": 3437.229736328125,
      "test/rot_rmse": 36.17388153076172,
      "test/trans_mae": 0.07007932662963867,
      "test/trans_mse": 0.024395354092121124,
      "test/trans_rmse": 0.08732880651950836
  },
  {
      "test/chamfer_distance": 0.09294154495000839,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.030197344720363617,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.049822807312012,
      "test/mat_f1": 0.04046022519469261,
      "test/mat_loss": 9.019625663757324,
      "test/mat_precision": 0.040460262447595596,
      "test/mat_recall": 0.040460262447595596,
      "test/n_critical_max": 128.55471801757812,
      "test/part_acc": 0.6433961987495422,
      "test/rot_mae": 31.983291625976562,
      "test/rot_mse": 3530.31591796875,
      "test/rot_rmse": 37.09149169921875,
      "test/trans_mae": 0.06845671683549881,
      "test/trans_mse": 0.02419455163180828,
      "test/trans_rmse": 0.08629211038351059
  },
  {
      "test/chamfer_distance": 0.10074230283498764,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.031229278072714806,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 8.51587963104248,
      "test/mat_f1": 0.03632359579205513,
      "test/mat_loss": 8.484649658203125,
      "test/mat_precision": 0.036323629319667816,
      "test/mat_recall": 0.036323629319667816,
      "test/n_critical_max": 130.8000030517578,
      "test/part_acc": 0.6367924809455872,
      "test/rot_mae": 31.351808547973633,
      "test/rot_mse": 3511.552734375,
      "test/rot_rmse": 36.79029846191406,
      "test/trans_mae": 0.07108917087316513,
      "test/trans_mse": 0.02587372623383999,
      "test/trans_rmse": 0.08943705260753632
  },
  {
      "test/chamfer_distance": 0.09827504307031631,
      "test/cls_acc": 1.0,
      "test/cls_f1": 1.0,
      "test/cls_loss": 0.030954325571656227,
      "test/cls_precision": 1.0,
      "test/cls_recall": 1.0,
      "test/loss": 9.000120162963867,
      "test/mat_f1": 0.03573187440633774,
      "test/mat_loss": 8.969165802001953,
      "test/mat_precision": 0.03573191538453102,
      "test/mat_recall": 0.03573191538453102,
      "test/n_critical_max": 136.5662841796875,
      "test/part_acc": 0.6628788113594055,
      "test/rot_mae": 31.026798248291016,
      "test/rot_mse": 3471.57421875,
      "test/rot_rmse": 36.397438049316406,
      "test/trans_mae": 0.06911768764257431,
      "test/trans_mse": 0.025298241525888443,
      "test/trans_rmse": 0.08667296916246414
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
