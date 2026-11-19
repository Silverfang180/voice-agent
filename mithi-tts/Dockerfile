# Use official Python 3.12 Linux base image
FROM python:3.12-slim

# Install system utilities needed for audio and build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install CPU-only PyTorch from official PyTorch CPU wheel index
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install official Mithi inference dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download official spaCy English model
RUN python -m spacy download en_core_web_sm

# Directory for model weights and output files
RUN mkdir -p /app/model /app/output

# Default environment configurations
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Default entrypoint for running TTS inference
CMD ["python", "--version"]

