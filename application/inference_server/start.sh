#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Jigsaw Inference Server"
echo "============================================"
echo "  Repo root:  $REPO_ROOT"
echo "  Server dir: $SCRIPT_DIR"
echo ""

# activate the conda environment
eval "$(conda shell.bash hook)"
conda activate assembly
echo "[OK] Conda environment 'assembly' activated"

# install inference-server-specific dependencies
echo ""
echo "[1/2] Installing inference server dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# start the FastAPI server
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PRELOAD="${PRELOAD_MODEL:-}"

echo ""
echo "[2/2] Starting uvicorn on $HOST:$PORT ..."
echo "  PRELOAD_MODEL=$PRELOAD"
echo ""

cd "$REPO_ROOT"
PRELOAD_MODEL="$PRELOAD" uvicorn \
    application.inference_server.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1
