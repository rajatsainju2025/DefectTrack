# CPU image by default. For GPU, build with:
#   docker build --build-arg BASE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime -t defecttrack:gpu .
ARG BASE=python:3.11-slim
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU PyTorch first (skipped automatically if the base image has torch).
RUN python -c "import torch" 2>/dev/null || \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY defecttrack ./defecttrack
RUN pip install --no-cache-dir .

COPY scripts ./scripts
COPY data ./data
RUN python scripts/download_weights.py

ENTRYPOINT ["defecttrack", "--weights", "/app/weights/best.pth"]
CMD ["data/video_sample/images", "--output", "/results"]
