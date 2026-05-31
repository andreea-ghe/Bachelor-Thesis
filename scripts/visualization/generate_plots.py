"""
Generate plots for the training and evaluation curves.
"""
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "plots"


TRAINING_CURVES = {
    # "Two Pieces Baseline Incomplete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_two_piece" / "jigsaw_finetune_everyday_2026-02-15-19-06-54" / "version_0" / "metrics.csv"))),
    "Two Pieces Baseline Complete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_two_piece_correct_ds" / "jigsaw_finetune_everyday_two_piece_correct_ds_2026-03-15-17-25-40" / "version_0" / "metrics.csv"))),
    # "Two Pieces Pair Attention Incomplete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_pair_attn" / "jigsaw_finetune_everyday_pair_attn_2026-02-18-17-34-42" / "version_0" / "metrics.csv"))),
    # "Two Pieces Pair Attention Complete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_pair_attn_correct_ds" / "jigsaw_finetune_everyday_pair_attn_correct_ds_2026-03-17-21-42-06" / "version_0" / "metrics.csv"))),
    # "Two Pieces Double Attention Incomplete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_double_attn_correct_ds" / "jigsaw_finetune_everyday_double_attn_correct_ds_2026-03-20-11-22-32" / "version_0" / "metrics.csv"))),
    # "Two Pieces Double Attention Complete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_double_attn_correct_ds_correct_params" / "jigsaw_finetune_everyday_double_attn_correct_ds_correct_params_2026-04-13-21-37-45" / "version_0" / "metrics.csv"))),
    # "Two Pieces Double Attention Incomplete Wrong Params": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_double_attn" / "jigsaw_finetune_everyday_new_arch_2026-03-06-22-04-25" / "version_0" / "metrics.csv"))),
    # "Multi Pieces Baseline Complete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_multi_everyday_correct_ds" / "jigsaw_finetune_multi_everyday_correct_ds_2026-03-26-07-28-12" / "version_0" / "metrics.csv"))),
    # "Multi Pieces Pair Attention Complete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_multi_everyday_pair_attn_correct_ds" / "jigsaw_finetune_multi_everyday_pair_attn_correct_ds_2026-03-28-04-04-22" / "version_0" / "metrics.csv"))),
    # "Multi Pieces Double Attention Complete": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_multi_everyday_double_attn_correct_ds" / "jigsaw_finetune_multi_everyday_double_attn_correct_ds_2026-03-30-06-03-46" / "version_0" / "metrics.csv"))),
    "Two Pieces Gabriel r=0 (no protection)": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_gabriel_two_piece_correct_ds" / "jigsaw_finetune_everyday_gabriel_two_piece_correct_ds_2026-05-17-21-37-24" / "version_0" / "metrics.csv"))),
    "Two Pieces Gabriel r=0.5 (50%)": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_gabriel_two_piece_correct_ds" / "jigsaw_finetune_everyday_gabriel_two_piece_correct_ds_2026-05-18-11-54-01" / "version_0" / "metrics.csv"))),
    "Two Pieces Gabriel r=0.75 (75%)": sorted(glob.glob(str(RESULTS_ROOT / "jigsaw_finetune_everyday_gabriel_two_piece_correct_ds" / "jigsaw_finetune_everyday_gabriel_two_piece_correct_ds_2026-05-20-07-53-44" / "version_0" / "metrics.csv"))),
}

EVAL_EXPERIMENTS = {}

BAR_METRICS = ["mat_f1", "part_acc", "rot_mae", "trans_mae", "chamfer_distance"]
BAR_METRIC_LABELS = {
    "mat_f1": "Matching F1",
    "part_acc": "Part Accuracy",
    "rot_mae": "Rotation MAE (°)",
    "trans_mae": "Translation MAE",
    "chamfer_distance": "Chamfer Distance",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
    "#56B4E9", "#E69F00", "#000000", "#999999", "#8B0000",

    "#1B9E77",  # teal-green
    "#E7298A",  # strong pink
    "#66A61E",  # lime green
    "#E6AB02",  # mustard
    "#A6761D",  # brown
    "#666666",  # dark gray
    "#1F78B4",  # alt blue
    "#B2DF8A",  # light green
    "#FB9A99",  # soft red
    "#CAB2D6"   # lavender
]

def savefig(fig, name: str):
    """Save figure"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    plt.close(fig)


def load_training_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_learning_rate(experiments: dict):
    """
    Learning rate schedule (both param groups).
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (name, csv_paths) in enumerate(experiments.items()):
        if not csv_paths:
            continue
        df = load_training_csv(csv_paths[0])
        lr_cols = [c for c in df.columns if c.startswith("lr-")]
        if not lr_cols:
            continue

        for j, col in enumerate(lr_cols):
            lr_data = df.dropna(subset=[col])
            if lr_data.empty:
                continue
            label_suffix = col.split("/")[-1] if "/" in col else col
            ax.plot(lr_data["step"], lr_data[col], label=f"{name} — {label_suffix}", color=COLORS[(i * 2 + j) % len(COLORS)], linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    fig.tight_layout()
    savefig(fig, "learning_rate_schedule")


def extract_val_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that contain validation metrics."""
    val_acc_col = next((c for c in df.columns if c == "val/cls_acc"), None)
    if val_acc_col:
        return df.dropna(subset=[val_acc_col]).copy()
    return pd.DataFrame()


def plot_val_losses(experiments: dict):
    """
    Validation loss (cls, mat, total) over epochs.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    loss_cols = ["val/cls_loss", "val/mat_loss", "val/rig_loss", "val/loss"]
    loss_titles = ["Val Segmentation Loss", "Val Matching Loss", "Val Rigidity Loss", "Val Total Loss"]

    for ax, col, title in zip(axes, loss_cols, loss_titles):
        for i, (name, csv_paths) in enumerate(experiments.items()):
            if not csv_paths:
                continue
            all_series = []
            for p in csv_paths:
                df = load_training_csv(p)
                val_df = extract_val_rows(df)
                if col in val_df.columns:
                    series = val_df[["epoch", col]].set_index("epoch")[col]
                    all_series.append(series)

            if not all_series:
                continue
            combined = pd.concat(all_series, axis=1)
            mean = combined.mean(axis=1)
            std = combined.std(axis=1)
            epochs = mean.index
            color = COLORS[i % len(COLORS)]
            ax.plot(epochs, mean, label=name, color=color, linewidth=1.5, marker="o", markersize=3)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Validation Loss over Epochs", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "validation_losses")


def plot_val_mat_f1(experiments: dict):
    """
    val/mat_f1 learning curves across experiments (the core result).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (name, csv_paths) in enumerate(experiments.items()):
        if not csv_paths:
            continue
        all_series = []
        for p in csv_paths:
            df = load_training_csv(p)
            val_df = extract_val_rows(df)
            if "val/mat_f1" in val_df.columns:
                series = val_df[["epoch", "val/mat_f1"]].set_index("epoch")["val/mat_f1"]
                all_series.append(series)

        if not all_series:
            continue
        combined = pd.concat(all_series, axis=1)
        mean = combined.mean(axis=1)
        std = combined.std(axis=1)
        epochs = mean.index
        color = COLORS[i % len(COLORS)]
        ax.plot(epochs, mean, label=name, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Matching F1")
    ax.set_title("Validation Matching F1 over Training")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "val_mat_f1_curves")


def plot_val_cls_acc(experiments: dict):
    """
    val/cls_acc stability (proves no catastrophic forgetting).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (name, csv_paths) in enumerate(experiments.items()):
        if not csv_paths:
            continue
        all_series = []
        for p in csv_paths:
            df = load_training_csv(p)
            val_df = extract_val_rows(df)
            if "val/cls_acc" in val_df.columns:
                series = val_df[["epoch", "val/cls_acc"]].set_index("epoch")["val/cls_acc"]
                all_series.append(series)

        if not all_series:
            continue
        combined = pd.concat(all_series, axis=1)
        mean = combined.mean(axis=1)
        std = combined.std(axis=1)
        epochs = mean.index
        color = COLORS[i % len(COLORS)]
        ax.plot(epochs, mean, label=name, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Segmentation Accuracy")
    ax.set_title("Validation Segmentation Accuracy (Catastrophic Forgetting Check)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.95)
    fig.tight_layout()
    savefig(fig, "val_cls_acc_stability")


def extract_train_epoch_means(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-batch training rows into per-epoch means."""
    train_cols = [c for c in df.columns if c.startswith("train/")]
    if not train_cols:
        return pd.DataFrame()
    first_col = next((c for c in train_cols if "loss" in c or "acc" in c), train_cols[0])
    subset = df.dropna(subset=[first_col]).copy()
    return subset.groupby("epoch")[train_cols].mean().reset_index()


def plot_training_losses(experiments: dict):
    """
    Training loss decomposition (cls_loss, mat_loss, total loss) over epochs.
    """
    loss_cols = ["train/cls_loss", "train/mat_loss", "train/rig_loss", "train/loss"]
    loss_titles = ["Segmentation Loss", "Matching Loss", "Rigidity Loss", "Total Loss"]

    for name, csv_paths in experiments.items():
        if not csv_paths:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)

        for p in csv_paths:
            df = load_training_csv(p)
            epoch_df = extract_train_epoch_means(df)
            if epoch_df.empty:
                continue

            for ax, col, title in zip(axes, loss_cols, loss_titles):
                if col not in epoch_df.columns:
                    continue
                ax.plot(epoch_df["epoch"], epoch_df[col],
                        linewidth=1.5, alpha=0.8)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"Training Losses — {name}", fontsize=14, y=1.02)
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("+", "plus")
        savefig(fig, f"training_losses_{safe_name}")


def parse_eval_log(path: str) -> dict:
    """
    Parse an eval_log file and return a dict.
    """
    metrics = {}
    with open(path, "r") as f:
        for line in f:
            if "test/cls_loss:" in line and "test/mat_f1:" in line:
                pairs = line.strip().split(";")
                for pair in pairs:
                    pair = pair.strip()
                    if ":" in pair:
                        key, val = pair.split(":", 1)
                        key = key.strip().replace("test/", "")
                        try:
                            metrics[key] = float(val.strip())
                        except ValueError:
                            pass
                break
    return metrics


def aggregate_eval_logs(paths: list) -> dict:
    """Parse multiple eval logs and return {metric: (mean, std, values)}."""
    all_metrics = defaultdict(list)
    for p in paths:
        m = parse_eval_log(p)
        for k, v in m.items():
            all_metrics[k].append(v)

    result = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        result[k] = (arr.mean(), arr.std(), arr)
    return result


def plot_eval_bar_charts(eval_experiments: dict, group_name: str = ""):
    """
    Grouped bar charts with error bars for each evaluation metric.
    """
    agg = {}
    for name, paths in eval_experiments.items():
        if paths:
            agg[name] = aggregate_eval_logs(paths)

    if not agg:
        print(f"  No eval data found for group '{group_name}', skipping.")
        return

    metrics_to_plot = [m for m in BAR_METRICS if any(m in a for a in agg.values())]
    if not metrics_to_plot:
        print(f"  No matching metrics in eval logs for group '{group_name}', skipping.")
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    names = list(agg.keys())
    x = np.arange(len(names))

    for ax, metric in zip(axes, metrics_to_plot):
        means = []
        stds = []
        for name in names:
            if metric in agg[name]:
                m, s, _ = agg[name][metric]
                means.append(m)
                stds.append(s)
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6, color=[COLORS[i % len(COLORS)] for i in range(len(names))], edgecolor="black", linewidth=0.5, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(BAR_METRIC_LABELS.get(metric, metric))
        ax.set_title(BAR_METRIC_LABELS.get(metric, metric))
        ax.grid(True, axis="y", alpha=0.3)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{m:.4f}", ha="center", va="bottom", fontsize=7)

    title = "Evaluation Metrics"
    if group_name:
        title += f" — {group_name}"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    suffix = f"_{group_name.lower().replace(' ', '_')}" if group_name else ""
    savefig(fig, f"eval_bar_charts{suffix}")


def plot_eval_comparison_mat_f1(eval_experiments: dict):
    """
    Single bar chart of mat_f1 across all configurations.
    """
    agg = {}
    for name, paths in eval_experiments.items():
        if paths:
            agg[name] = aggregate_eval_logs(paths)

    if not agg:
        return

    names = list(agg.keys())
    means = []
    stds = []
    for name in names:
        if "mat_f1" in agg[name]:
            m, s, _ = agg[name]["mat_f1"]
            means.append(m)
            stds.append(s)
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55, color=[COLORS[i % len(COLORS)] for i in range(len(names))], edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Matching F1")
    ax.set_title("Matching F1 Across All Configurations")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "eval_mat_f1_comparison")


def main():
    has_training = any(paths for paths in TRAINING_CURVES.values())
    if has_training:
        plot_learning_rate(TRAINING_CURVES)
        plot_val_losses(TRAINING_CURVES)
        plot_val_mat_f1(TRAINING_CURVES)
        plot_val_cls_acc(TRAINING_CURVES)
        plot_training_losses(TRAINING_CURVES)
    else:
        print("No training CSVs found, skipping plots.")

    has_eval = any(paths for paths in EVAL_EXPERIMENTS.values())
    if has_eval:
        two_piece = {k: v for k, v in EVAL_EXPERIMENTS.items() if k.startswith("Two Pieces")}
        multi = {k: v for k, v in EVAL_EXPERIMENTS.items() if k.startswith("Multi Pieces")}

        if two_piece:
            plot_eval_bar_charts(two_piece, "Two-Piece")
        if multi:
            plot_eval_bar_charts(multi, "Multi-Piece")

        plot_eval_comparison_mat_f1(EVAL_EXPERIMENTS)
    else:
        print("No eval logs found, skipping plots.")


if __name__ == "__main__":
    main()
