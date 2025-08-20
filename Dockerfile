# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    wget \
    git \
    build-essential \
    # PDF processing
    poppler-utils \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-equ \
    # Image processing
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # OpenCV dependencies
    libglib2.0-0 \
    libgl1-mesa-glx \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install PDFFigures2 (Java-based)
RUN apt-get update && apt-get install -y default-jre && \
    wget https://github.com/allenai/pdffigures2/releases/download/v0.1.0/pdffigures2-0.1.0.jar -O /usr/local/bin/pdffigures2.jar && \
    echo '#!/bin/bash\njava -jar /usr/local/bin/pdffigures2.jar "$@"' > /usr/local/bin/pdffigures2 && \
    chmod +x /usr/local/bin/pdffigures2

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p paper logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GROBID_URL=http://grobid:8070

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]