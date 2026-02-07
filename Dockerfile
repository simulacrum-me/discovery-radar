FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Create directory for model cache
RUN mkdir -p pretrained_models

# Expose no ports - MCP uses stdio transport
# Models are cached in /app/pretrained_models

CMD ["python", "server.py"]
