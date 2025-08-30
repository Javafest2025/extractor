# Multi-stage build for PDF Extractor FastAPI Service
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-prod.txt requirements.txt

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU-only first, then other requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

# Install runtime dependencies only (no dev packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    # OpenCV dependencies (runtime only)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Image processing dependencies (runtime only)
    libpng16-16 \
    libjpeg62-turbo \
    libtiff6 \
    # Additional system libraries
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user
RUN addgroup --system app && adduser --system --no-create-home --ingroup app app

# Create necessary directories with proper permissions
RUN mkdir -p /app/cache /app/paper /app/logs /home/app/.cache && \
    chown -R app:app /app /home/app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables for HuggingFace cache
ENV TRANSFORMERS_CACHE="/app/cache/transformers"
ENV HF_HOME="/app/cache/huggingface"
ENV TORCH_HOME="/app/cache/torch"

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create necessary directories and set permissions
RUN mkdir -p ./paper/uploads ./paper/results ./cache/transformers ./cache/huggingface ./cache/torch ./logs && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8002

# Health check (using the simple health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
