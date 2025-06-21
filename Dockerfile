# Use CUDA 12.2 base image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Gunicorn
RUN python3 -m pip install --upgrade pip
RUN pip install gunicorn

# Install torch and torchvision (CUDA 12.2 builds)
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose Flask port (use 6001 for consistency)
EXPOSE 8001

# Health check with startup grace period
#HEALTHCHECK --interval=30s --timeout=20s --start-period=60s --retries=3 \
 #   CMD curl -f http://localhost:6001/health || exit 1

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "4", "--timeout", "120", "app:app"]
