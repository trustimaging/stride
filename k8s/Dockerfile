# Dockerfile for Mosaic on Kubernetes
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    minio \
    uvloop \
    pyzmq \
    cloudpickle \
    psutil \
    zict \
    cached_property \
    h5py

# Copy source code
COPY mosaic/ /app/mosaic/
COPY stride/ /app/stride/
COPY setup.py /app/
COPY k8s_inversion.py /app/

# Install the packages
RUN pip install -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["python"]
