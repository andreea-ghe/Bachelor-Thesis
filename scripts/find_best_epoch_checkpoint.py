import csv
import sys


def extract_val_metrics(csv_path):
    key_cols = [
        'epoch', 'val/mat_f1', 'val/loss', 'val/cls_f1',
        'val/mat_loss', 'val/rig_loss', 'val/cls_acc',
        'val/mat_precision', 'val/mat_recall', 'val/n_critical_max',
    ]

    val_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        available = [c for c in key_cols if c in reader.fieldnames]
        for row in reader:
            if row.get('val/mat_f1') and row['val/mat_f1'].strip():
                val_rows.append({k: row.get(k, '') for k in available})

    if not val_rows:
        print("No validation rows found.")
        return

    best_idx = max(range(len(val_rows)),
                   key=lambda i: float(val_rows[i].get('val/mat_f1', 0)))

    header = f"{'Epoch':>5} | {'mat_f1':>8} | {'val_loss':>9} | {'cls_f1':>8} | {'mat_loss':>9} | {'rig_loss':>9}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(val_rows):
        print(
            f"{r.get('epoch', '?'):>5} | "
            f"{float(r.get('val/mat_f1', 0)):>8.4f} | "
            f"{float(r.get('val/loss', 0)):>9.4f} | "
            f"{float(r.get('val/cls_f1', 0)):>8.4f} | "
            f"{float(r.get('val/mat_loss', 0)):>9.4f} | "
            f"{float(r.get('val/rig_loss', 0)):>9.4f}"
        )

    best = val_rows[best_idx]
    print(f"\nBest epoch: {best['epoch']}  (mat_f1 = {float(best['val/mat_f1']):.4f})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/metrics.csv>")
        sys.exit(1)
    extract_val_metrics(sys.argv[1])
