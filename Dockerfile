# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Satellite Defence Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for OpenCV, rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    gdal-bin libgdal-dev git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY pipeline.py .
COPY pipeline_config.yaml .
COPY Makefile .

# Default: run full pipeline
ENTRYPOINT ["python", "pipeline.py"]
CMD ["--stage", "all"]
