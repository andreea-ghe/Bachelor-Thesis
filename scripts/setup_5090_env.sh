#!/bin/bash
set -e

ENV_NAME="assembly"

echo "============================================"
echo "  RTX 5090 Environment Setup (CUDA 12.8)"
echo "============================================"

# Step 1: Create conda environment with Python 3.10
echo ""
echo "[1/6] Creating conda environment with Python 3.10..."
conda create -n $ENV_NAME python=3.10.14 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Step 2: Install PyTorch 2.7 with CUDA 12.8 (required for sm_120)
echo ""
echo "[2/6] Installing PyTorch 2.7.0 + CUDA 12.8..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA support
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')"

# Step 3: Install PyG (PyTorch Geometric) with matching CUDA wheels
echo ""
echo "[3/6] Installing PyTorch Geometric 2.7.0..."
pip install torch_geometric==2.7.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Step 4: Install pytorch3d from source (no prebuilt wheels for PyTorch 2.7)
echo ""
echo "[4/6] Installing pytorch3d from source (this may take a few minutes)..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Step 5: Install PyTorch Lightning and core ML packages
echo ""
echo "[5/6] Installing PyTorch Lightning and ML packages..."
pip install pytorch-lightning==2.5.0 torchmetrics==1.6.0 lightning-utilities==0.11.9

# Step 6: Install remaining dependencies
echo ""
echo "[6/6] Installing remaining dependencies..."
pip install \
    numpy==1.26.4 \
    scipy==1.13.1 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    matplotlib==3.9.3 \
    pillow==10.4.0 \
    open3d==0.19.0 \
    gtsam==4.2 \
    einops==0.8.0 \
    trimesh==4.5.3 \
    pyyaml==6.0.2 \
    tqdm==4.67.1 \
    yacs==0.1.8 \
    easydict \
    addict==2.4.0 \
    plyfile==0.7.4 \
    tabulate==0.9.0 \
    networkx==2.8.8 \
    fsspec \
    filelock \
    packaging

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
python -c "
import torch
import torch_geometric
import pytorch3d
import pytorch_lightning
import open3d
print(f'PyTorch:          {torch.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
print(f'CUDA version:     {torch.version.cuda}')
print(f'PyG:              {torch_geometric.__version__}')
print(f'PyTorch3D:        {pytorch3d.__version__}')
print(f'Lightning:        {pytorch_lightning.__version__}')
print(f'Open3D:           {open3d.__version__}')
"
