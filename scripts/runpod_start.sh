#!/bin/bash
# Trivima — RunPod setup & test
#
# Usage:
#   bash scripts/runpod_start.sh
#
# After setup:
#   python trivima/app.py --image room.jpg --stats
#   python trivima/app.py --image room.jpg --save-grid room.npz --export-ply room.ply
#   python trivima/app.py --image room.jpg --render-preview preview.png
#
# RunPod template: PyTorch 2.1+ / CUDA 12.1
# Minimum: A100 40GB

set -e

echo "============================================"
echo "  Trivima — RunPod Setup"
echo "============================================"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: No GPU"
echo "  Python: $(python3 --version 2>&1)"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch: not installed"
echo ""

# Dependencies
echo "Installing dependencies..."
pip install -q numpy scipy scikit-image open3d \
    opencv-python opencv-contrib-python \
    msgpack Pillow transformers timm 2>/dev/null
echo "  Done."

# Perception models
echo ""
bash scripts/setup_models.sh

# Validate with synthetic test
echo ""
echo "============================================"
echo "  Running synthetic validation..."
echo "============================================"
python3 scripts/run_demo.py --synthetic

echo ""
echo "============================================"
echo "  Ready."
echo ""
echo "  Process a photo:"
echo "    python3 trivima/app.py --image room.jpg --stats"
echo ""
echo "  Save outputs:"
echo "    python3 trivima/app.py --image room.jpg \\"
echo "      --save-grid room.npz \\"
echo "      --export-ply room.ply \\"
echo "      --render-preview preview.png"
echo "============================================"
