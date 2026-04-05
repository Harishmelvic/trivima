FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libglfw3-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# PyTorch with CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Python dependencies
RUN pip install \
    numpy scipy scikit-image \
    open3d opencv-python opencv-contrib-python \
    moderngl "moderngl-window[glfw,imgui]" \
    msgpack Pillow transformers timm lpips \
    nanobind scikit-build-core

# Copy project
WORKDIR /workspace/trivima
COPY . .

# Setup models and build
RUN bash scripts/setup_models.sh

EXPOSE 8888

# Default: run the demo
CMD ["python", "scripts/run_demo.py", "--synthetic"]
