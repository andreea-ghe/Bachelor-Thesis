# 3D Object Reassembly of Broken Objects Using Deep Learning Methods

**Bachelor's Thesis** — Babeș-Bolyai University, Faculty of Mathematics and Computer Science  
**Author:** Andreea-Ioana Gheorghe  
**Supervisors:** Dr. Anca Andreica (University Professor), Raluca-Diana Chiș (PhD candidate)  
**Specialization:** Artificial Intelligence

This repository contains the codebase for the bachelor's thesis *"3D Object Reassembly of Broken Objects using Deep Learning Methods"*, which investigates targeted extensions to the [Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw) framework for multi-piece 3D fracture assembly.

---

## Overview

Given point clouds of fractured object pieces, the goal is to predict the 6-DoF rigid transformation (rotation + translation) that maps each piece back to its original position. This thesis builds on the Jigsaw pipeline — the first end-to-end learning-based framework for multi-piece fracture assembly — and proposes four extensions:

1. **Pair Geometric Attention Bias** — Adapted from [GPAT](https://arxiv.org/abs/2310.13520), encodes pairwise distances and angles between fragments via Gaussian RBFs and injects them as scalar biases into cross-attention. Within Jigsaw's preprocessing context, it acts as a learned soft regularizer that smooths attention distributions, consistently improving out-of-distribution generalization.

2. **Gated Double Attention** — Adds a second round of self-attention and cross-attention through learned gates (ReZero initialization), providing +2 to 5 percentage points under data scarcity and consistent cross-domain improvements.

3. **Gabriel Graph Neighborhood Filtering** — Replaces standard kNN grouping in PointNet++ with a Gabriel graph filter that preserves only surface-faithful neighbors, improving cross-domain pose quality on artifact objects at the cost of slightly fewer correspondences.

4. **ICP Refinement** — Inserts an Iterative Closest Point step between RANSAC-based pairwise pose estimation and Shonan global synchronization, adding ~1 percentage point across all configurations.

All experiments are conducted on the [Breaking Bad](https://breaking-bad-dataset.github.io/) dataset (everyday + artifact subsets), with models finetuned on both two-piece and multi-piece (2–4) fracture patterns. An interactive web application accompanies the thesis for visual comparison of assembly results across 11 model variants.

---

## Repository Structure

```
bachelor-thesis/
├── feature_extractor/          # PointNet++ backbone, attention mechanisms, pair encoder, Gabriel filter
│   ├── pointnet_architecture.py    # PointNet++ multi-scale encoder-decoder
│   ├── attention_mechanisms.py     # Self-attention, cross-attention, Point Transformer
│   ├── pair_geometric_encoder.py   # Geometric pair bias (RBF-encoded distances/angles)
│   ├── utils_encoder_decoder.py    # PointNet encoder, Gabriel graph filtering
│   └── utils_gaussian_rbf.py       # Gaussian radial basis functions
│
├── jigsaw_pipeline/            # End-to-end Jigsaw model and losses
│   ├── joint_segmentation_align_model.py   # Main model: encoder → attention → seg → match → align
│   ├── utils_losses.py                     # Segmentation, permutation, and rigidity losses
│   └── utils_pairwise_alignment.py         # Weighted Horn alignment from soft correspondences
│
├── surface_segmentation/       # Fracture vs. intact point classification
│   └── segmentation_classifier.py          # MLP classification head (binary or multi-class)
│
├── multipart_matching/         # Correspondence establishment
│   ├── affinity.py             # Primal-dual descriptor for anti-self-matching
│   ├── utils_sinkhorn.py       # Differentiable optimal transport (soft matching)
│   └── utils_hungarian.py      # Discrete assignment at inference
│
├── global_alignment/           # Pose recovery from pairwise matches
│   ├── utils_alignment.py      # RANSAC + optional ICP refinement
│   ├── utils_shonan.py         # Shonan rotation averaging (GTSAM)
│   ├── spanning_tree_alignment.py  # MST fallback when Shonan fails
│   └── utils_pose_graph.py     # Minimum spanning tree construction
│
├── base_pipeline/              # PyTorch Lightning training infrastructure
│   ├── base_model.py           # LightningModule: train/val/test steps, optimizer, metrics
│   ├── utils_evaluation.py     # Part accuracy, Chamfer distance, rotation/translation MAE
│   ├── utils_lr_scheduler.py   # Cosine annealing with warmup restarts
│   └── utils_optimizer.py      # Parameter-group-specific learning rates
│
├── dataset_preprocessing/      # Breaking Bad dataset loading
│   ├── fracture_assembly_dataset.py    # Multi-piece loader (2–20 pieces, area sampling, SO(3) augmentation)
│   ├── fracture_pairs_dataset.py       # Two-piece pair loader with CSV splits
│   └── dataset_config.py               # Paths, categories, sampling parameters
│
├── utilities/                  # Cross-cutting utilities
│   ├── utils_config.py         # Global CONFIG (EasyDict + YAML merging)
│   ├── utils_parse_args.py     # CLI argument parsing
│   ├── rotation.py             # Rotation3D wrapper (quaternion/matrix/axis-angle)
│   └── utils_stdout.py         # Stdout logging to file
│
├── experiments/                # Training & evaluation entry points + experiment configs
│   ├── train_model.py          # Training with checkpointing, CSV logging, loss scheduling
│   ├── eval_model.py           # Test-set evaluation
│   ├── model_config.py         # Default model hyperparameters
│   ├── original_scripts/       # Baseline Jigsaw (binary/multi)
│   ├── two_piece_scripts/      # Two-piece baseline finetuning
│   ├── pair_attn_scripts/      # Pair geometric attention
│   ├── double_attn_scripts/    # Gated double attention
│   ├── pair_double_attn_scripts/       # Combined pair + double attention
│   ├── gabriel_scripts/        # Gabriel graph filtering (50%/75% retention)
│   ├── multi_piece_scripts/    # Multi-piece (2–4) baseline
│   ├── multi_pair_attn_scripts/        # Multi-piece + pair attention
│   ├── multi_double_attn_scripts/      # Multi-piece + double attention
│   ├── multi_pair_double_attn_scripts/ # Multi-piece + both extensions
│   └── volume_constrained_scripts/     # Volume-constrained training subset
│
├── scripts/                    # Analysis, visualization, and tooling
│   ├── setup/                  # Environment setup (setup_5090_env.sh)
│   ├── inference/              # CLI inference, precomputation, thumbnail generation
│   ├── evaluation/             # Result extraction, checkpoint selection, consistency checks
│   ├── analysis/               # Pair encoder / double attention learning verification
│   └── visualization/          # Training curves, 3D transforms, Gabriel neighborhoods
│
├── application/                # Interactive web demo
│   ├── web/                    # Django frontend (Three.js 3D viewer, thesis slides)
│   └── inference_server/       # FastAPI GPU backend (11 model variants)
│
├── thesis/                     # LaTeX source for the bachelor's thesis
│   ├── main.tex
│   ├── chapter1_introduction.tex – chapter8_conclusions.tex
│   ├── appendix_application.tex
│   ├── references.bib
│   └── figures/
│
├── scss/                       # IEEE conference paper (10-page condensed version)
│
├── environment.yaml            # Conda env: Python 3.8 + CUDA 11.3 (original)
└── 5090_environment.yaml       # Conda env: Python 3.10 + PyTorch 2.7 + CUDA 12.8
```

---

## Installation

### Environment Setup

The codebase was originally developed with Python 3.8 / PyTorch 1.10 / CUDA 11.3 and has been migrated to Python 3.10 / PyTorch 2.7 / CUDA 12.8 for current-generation hardware.

**Option A — Original environment (CUDA 11.3):**

```bash
conda env create -f environment.yaml
conda activate assembly
```

**Option B — Modern environment (CUDA 12.8, recommended for RTX 4090/5090):**

```bash
bash scripts/setup/setup_5090_env.sh
conda activate assembly
```

This script installs PyTorch 2.7, PyTorch Geometric, pytorch3d (compiled from source), PyTorch Lightning, Open3D, and GTSAM.

### Web Application Dependencies

```bash
pip install -r application/web/requirements.txt
pip install -r application/inference_server/requirements.txt
```

---

## Dataset

All experiments use the [Breaking Bad Dataset](https://breaking-bad-dataset.github.io/). Please use the updated inner-face-free version.

After downloading and processing, the expected data layout is:

```
data/
└── breaking_bad/
    ├── everyday/               # 20 daily object categories (407 train / 91 val objects)
    │   ├── BeerBottle/
    │   ├── Bottle/
    │   ├── Bowl/
    │   └── ...
    ├── artifact/               # Archaeological-style objects (40 objects, cross-domain eval)
    ├── everyday.train.txt
    ├── everyday.val.txt
    ├── artifact.train.txt
    └── artifact.val.txt
```

- **Everyday** — 34,075 training / 7,679 validation fracture patterns across 20 categories. Used for all training.
- **Artifact** — 3,651 fracture patterns from sculptures and archaeological objects. Used exclusively for cross-domain evaluation (no finetuning on artifact data).

---

## Training

```bash
python -m experiments.train_model --cfg experiments/<config_dir>/<config>.yaml
```

Example — finetune with gated double attention on two-piece everyday data:

```bash
python -m experiments.train_model --cfg experiments/double_attn_scripts/finetune_everyday.yaml
```

Training produces checkpoints and CSV logs in `results/<MODEL_NAME>/`.

### Experiment Configurations

Each experiment directory contains YAML configs for different evaluation scenarios:

| Directory | Extension | Pieces |
|-----------|-----------|--------|
| `two_piece_scripts/` | Baseline | 2 |
| `pair_attn_scripts/` | Pair attention | 2 |
| `double_attn_scripts/` | Double attention | 2 |
| `pair_double_attn_scripts/` | Pair + double | 2 |
| `gabriel_scripts/` | Gabriel filtering (50%/75%) | 2 |
| `multi_piece_scripts/` | Baseline | 2–4 |
| `multi_pair_attn_scripts/` | Pair attention | 2–4 |
| `multi_double_attn_scripts/` | Double attention | 2–4 |
| `multi_pair_double_attn_scripts/` | Pair + double | 2–4 |

Key YAML fields:

```yaml
MODEL_NAME: experiment_name
DATASET: breaking_bad.all_piece_matching

DATA:
  SUBSET: everyday
  MAX_NUM_PART: 20
  NUM_PC_POINTS: 5000

MODEL:
  USE_PAIR_BIAS: false       # enable pair geometric attention
  USE_DOUBLE_ATTN: false     # enable gated double attention
  USE_GABRIEL: false          # enable Gabriel graph filtering
  GABRIEL_MIN_KEEP_RATIO: 0.5 # minimum neighbor retention ratio

WEIGHT_FILE: checkpoint/pretrained.ckpt
```

---

## Evaluation

```bash
python -m experiments.eval_model --cfg experiments/<config_dir>/<config>.yaml
```

**Metrics:**

| Metric | Description |
|--------|-------------|
| Part Accuracy (PA ↑) | Fraction of fragments with Chamfer distance < 0.01 |
| Chamfer Distance (CD ↓) | Average bidirectional distance (×10⁻³) between predicted and ground-truth point clouds |
| Rotation MAE (↓) | Mean absolute rotation error in degrees |
| Translation MAE (↓) | Mean absolute translation error |
| Matching F1 (↑) | F1 score for correspondence predictions |

Results are written to `results/<MODEL_NAME>/`.

---

## Web Application

An interactive web demo lets users select a fractured object example, choose one of 11 model variants, and view the 3D assembly result compared to the input.

### Architecture

- **Frontend:** Django 6 + Tailwind CSS + Three.js (r170) — single-page thesis presentation with an interactive 3D demo panel
- **Backend:** FastAPI + Uvicorn — GPU inference server that loads model variants on demand

### Running the Demo

**1. Start the inference server** (requires GPU + model weights):

```bash
cd application/inference_server
bash start.sh
```

**2. Start the Django frontend:**

```bash
cd application/web
cp .env.example .env  # configure SECRET_KEY and GPU_POD_URL
python manage.py runserver
```

The web app also supports a **precomputed fallback mode**: if the GPU server is unreachable, it serves cached results from `precomputed_results.json` (tracked via Git LFS).

### Model Variants

The demo supports 11 model variants:

| Variant | Pieces | Extension |
|---------|--------|-----------|
| Jigsaw Baseline | 2 | — |
| Jigsaw Baseline (reduced) | 2 | — |
| Pair Attention | 2 | Geometric pair bias |
| Pair Attention (reduced) | 2 | Geometric pair bias |
| Double Attention | 2 | Gated double attention |
| Double Attention (reduced) | 2 | Gated double attention |
| Gabriel 50% | 2 | Gabriel filtering (50% retention) |
| Gabriel 75% | 2 | Gabriel filtering (75% retention) |
| Baseline | 2–4 | — |
| Pair Attention | 2–4 | Geometric pair bias |
| Double Attention | 2–4 | Gated double attention |

---

## Key Results

Experiments are conducted on the everyday validation split (in-domain) and artifact split (cross-domain, out-of-distribution).

**Two-piece assembly (everyday validation):**

- Pair attention acts as a learned soft regularizer, neutral on in-domain data but up to +2.2 PA on artifact objects
- Gated double attention provides +2–5 PA under data scarcity
- Gabriel filtering improves cross-domain pose quality (CD −10%, translation MAE −5%) at a slight cost in correspondence count
- ICP refinement adds ~1 PA consistently across all configurations

**Data scaling finding:** The full dataset yields over 10 percentage points more than the reduced (~63%) subset, confirming that data quantity compounds with architectural capacity.

---

## Utility Scripts

```bash
# precompute web demo results for all model variants
python -m scripts.inference.precompute_results

# generate thumbnail images for the demo examples
python -m scripts.inference.generate_thumbnails

# find best-performing examples for demos
python -m scripts.inference.find_best_examples

# plot training curves
python -m scripts.visualization.generate_plots

# verify that pair encoder / double attention layers are learning
python -m scripts.analysis.check_pair_encoder_learning
python -m scripts.analysis.check_double_attn_learning
```

---

## Configuration System

The configuration system merges three layers:

1. **`experiments/model_config.py`** — Default model hyperparameters (architecture flags, loss weights, dimensions)
2. **`dataset_preprocessing/dataset_config.py`** — Default dataset parameters (paths, categories, sampling)
3. **YAML config files** — Experiment-specific overrides

Loaded via `utilities/utils_config.py` and accessed as a global `CONFIG` EasyDict throughout the codebase.

---

## Thesis Document

The LaTeX source for the thesis is in `thesis/`. Chapter structure:

| Chapter | Topic |
|---------|-------|
| 1 | Introduction |
| 2 | Problem Definition (geometric vs. semantic assembly) |
| 3 | Theoretical Background (PointNet, PointNet++, attention, optimal transport) |
| 4 | Related Work (PMTR, PHFormer, GPAT, GARF, Jigsaw++) |
| 5 | Methodology (four proposed extensions) |
| 6 | Experiments and Results |
| 7 | Interactive Web Application |
| 8 | Conclusions, Threats to Validity, and Future Work |

A condensed 10-page IEEE conference paper is in `scss/`.

---

## Acknowledgement

This work builds on the [Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw) repository by Lu et al. The pair geometric attention bias is adapted from [GPAT](https://github.com/Fzaero/GPAT) by Li et al.
