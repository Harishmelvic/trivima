#!/bin/bash
# Trivima — Download all model checkpoints and dependencies
# Run this once on the cloud instance before first use.
#
# Usage: bash scripts/setup_models.sh
# Estimated download: ~8GB
# Estimated time: 5-10 minutes on a fast connection

set -e

CHECKPOINT_DIR="data/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================"
echo "Trivima Model Setup"
echo "============================================"

# --- 1. Depth Pro (Apple) ---
echo ""
echo "[1/4] Setting up Depth Pro..."
if [ ! -d "third_party/ml-depth-pro" ]; then
    git clone https://github.com/apple/ml-depth-pro.git third_party/ml-depth-pro
    cd third_party/ml-depth-pro
    pip install -e .
    # Download checkpoint
    python -c "import depth_pro; depth_pro.create_model_and_transforms(device='cpu')" 2>/dev/null || \
        echo "  Note: Depth Pro weights will download on first inference"
    cd ../..
else
    echo "  Already cloned."
fi

# --- 2. SAM 2 + Grounding DINO (fallback for SAM 3) ---
echo ""
echo "[2/4] Setting up SAM 2 + Grounding DINO..."
if [ ! -d "third_party/sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git third_party/sam2
    cd third_party/sam2
    pip install -e .
    cd ../..
fi

# Download SAM 2 checkpoint
SAM2_CKPT="$CHECKPOINT_DIR/sam2_hiera_large.pt"
if [ ! -f "$SAM2_CKPT" ]; then
    echo "  Downloading SAM 2 Hiera Large checkpoint..."
    wget -q --show-progress -O "$SAM2_CKPT" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
else
    echo "  SAM 2 checkpoint exists."
fi

# Grounding DINO
if [ ! -d "third_party/GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
    cd third_party/GroundingDINO
    pip install -e .
    cd ../..
fi

GDINO_CKPT="$CHECKPOINT_DIR/groundingdino_swint_ogc.pth"
if [ ! -f "$GDINO_CKPT" ]; then
    echo "  Downloading Grounding DINO checkpoint..."
    wget -q --show-progress -O "$GDINO_CKPT" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
else
    echo "  Grounding DINO checkpoint exists."
fi

# --- 3. DUSt3R (optional, for multi-image) ---
echo ""
echo "[3/4] Setting up DUSt3R (optional, for multi-image input)..."
if [ ! -d "third_party/dust3r" ]; then
    git clone --recursive https://github.com/naver/dust3r.git third_party/dust3r
    cd third_party/dust3r
    pip install -e .
    cd ../..
    echo "  DUSt3R installed. Checkpoint downloads on first use."
else
    echo "  Already cloned."
fi

# --- 4. Python dependencies ---
echo ""
echo "[4/4] Installing Python dependencies..."
pip install -q \
    numpy \
    torch torchvision torchaudio \
    open3d \
    opencv-python \
    opencv-contrib-python \
    moderngl \
    "moderngl-window[glfw,imgui]" \
    msgpack \
    Pillow \
    scipy \
    scikit-image \
    transformers \
    timm \
    lpips \
    2>/dev/null

# Build C++/CUDA core (if CMake + CUDA available)
echo ""
echo "Building Trivima native core..."
if command -v cmake &> /dev/null && command -v nvcc &> /dev/null; then
    pip install nanobind scikit-build-core
    pip install -e . 2>/dev/null || echo "  Note: Native build failed. Python-only mode will be used."
else
    echo "  CMake or CUDA not found. Skipping native build — Python-only mode."
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Quick test:"
echo "  python scripts/run_demo.py --image data/sample_images/room1.jpg"
echo ""
echo "Full app:"
echo "  python trivima/app.py --image data/sample_images/room1.jpg"
echo "============================================"
