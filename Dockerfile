FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace/trivima

# Install Python deps first (cached layer — only rebuilds if requirements change)
RUN pip install --no-cache-dir \
    runpod \
    numpy scipy scikit-image Pillow \
    opencv-python-headless opencv-contrib-python-headless

# Copy project code
COPY . .

# Entry point for RunPod serverless
CMD ["python3", "-u", "handler.py"]
