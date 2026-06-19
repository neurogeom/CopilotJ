# syntax=docker/dockerfile:1.7

# GPU-enabled Dockerfile for ChatImageJ Python Environment
# Multi-stage build with NVIDIA CUDA support

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM nvidia/cuda:13.3.0-devel-ubuntu22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    UV_CACHE_DIR=/root/.cache/uv

# Install Python 3.12 with the minimum builder dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN uv venv /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --active --frozen --no-dev --no-install-project

# =============================================================================
# Stage 2: Runtime - Final image
# =============================================================================
FROM nvidia/cuda:13.3.0-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 with the minimum runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code and runtime resources
COPY copilotj/ ./copilotj/
COPY templates/ ./templates/
COPY assets/ ./assets/
COPY knowledge_bank/ ./knowledge_bank/
COPY pyproject.toml README.md ./

# Expose the bridge server port
EXPOSE 8786

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8786/api/ping', timeout=5).read()" || exit 1

# Default entrypoint: bridge server
CMD ["python", "-m", "copilotj.server", "--host", "0.0.0.0", "--port", "8786"]
