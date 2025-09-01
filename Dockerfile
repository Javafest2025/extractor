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
    wget \
    curl \
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

# Set PyTorch to use only 1 thread to prevent CPU overload
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Stage 2: Runtime image
FROM python:3.11-slim

# Install runtime dependencies only (no dev packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
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
    # Java for PDFFigures2 (if needed)
    default-jre \
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

# Set environment variables for HuggingFace cache and EasyOCR
ENV HF_HOME="/app/cache/huggingface"
ENV TORCH_HOME="/app/cache/torch"
ENV EASYOCR_MODULE_PATH="/app/cache/easyocr"

# CPU optimization: Limit threads to prevent overload
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV BLAS_NUM_THREADS=1
ENV LAPACK_NUM_THREADS=1

# PyTorch CPU optimization
ENV TORCH_NUM_THREADS=1
ENV TORCH_NUM_INTEROP_THREADS=1

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create necessary directories and set permissions
RUN mkdir -p ./paper/uploads ./paper/results ./paper/figures ./paper/tables ./paper/json ./paper/text/sections ./paper/text/subsections ./paper/code ./paper/ocr_math ./cache/huggingface ./cache/torch ./cache/easyocr ./logs && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8002

# Health check (using the simple health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application with CPU optimization
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1", "--limit-concurrency", "10", "--limit-max-requests", "1000"]
