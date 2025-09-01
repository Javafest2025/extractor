# Multi-stage build for PDF Extractor FastAPI Service (Optimized for Memory)
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-prod.txt requirements-prod.txt

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install lightweight requirements only
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Stage 2: Runtime image
FROM python:3.11-slim

# Install runtime dependencies only (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    # Tesseract OCR (essential)
    tesseract-ocr \
    tesseract-ocr-eng \
    # Basic image processing
    libpng16-16 \
    libjpeg62-turbo \
    libtiff6 \
    # System libraries
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user
RUN addgroup --system app && adduser --system --no-create-home --ingroup app app

# Create necessary directories with proper permissions
RUN mkdir -p /app/cache /app/paper /app/logs && \
    chown -R app:app /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables for cache
ENV HF_HOME="/app/cache/huggingface"
ENV TORCH_HOME="/app/cache/torch"

# Memory optimization: Limit threads and processes
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV BLAS_NUM_THREADS=1
ENV LAPACK_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create necessary directories and set permissions
RUN mkdir -p ./paper/uploads ./paper/results ./paper/figures ./paper/tables ./paper/json ./paper/text/sections ./paper/text/subsections ./paper/code ./paper/ocr_math ./cache/huggingface ./cache/torch ./logs && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run with memory optimization
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1", "--limit-concurrency", "5", "--limit-max-requests", "500"]
