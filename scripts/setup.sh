#!/bin/bash

# setup.sh - Setup script for PDF Extractor

echo "==================================="
echo "PDF Extractor Setup Script"
echo "==================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python $required_version or higher is required. Found: Python $python_version"
    exit 1
fi
echo "✓ Python $python_version found"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Python dependencies installed"

# Download spaCy model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create necessary directories
echo "Creating project directories..."
mkdir -p paper
mkdir -p paper/uploads
mkdir -p paper/results
mkdir -p paper/figures
mkdir -p paper/tables
mkdir -p paper/code
mkdir -p paper/ocr_math
mkdir -p logs
echo "✓ Directories created"

# Check for system dependencies
echo "Checking system dependencies..."

# Check Tesseract
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract OCR found: $(tesseract --version | head -n1)"
else
    echo "⚠ Tesseract OCR not found. Please install: sudo apt-get install tesseract-ocr"
fi

# Check Java (for PDFFigures2)
if command -v java &> /dev/null; then
    echo "✓ Java found: $(java -version 2>&1 | head -n1)"
else
    echo "⚠ Java not found. PDFFigures2 requires Java. Please install: sudo apt-get install default-jre"
fi

# Check Docker (for GROBID)
if command -v docker &> /dev/null; then
    echo "✓ Docker found: $(docker --version)"
    
    # Start GROBID service
    echo "Starting GROBID service with Docker..."
    docker-compose up -d grobid
    
    # Wait for GROBID to be ready
    echo "Waiting for GROBID to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8070/api/isalive > /dev/null 2>&1; then
            echo "✓ GROBID service is running"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "⚠ GROBID service failed to start. Please check Docker logs."
        fi
        sleep 2
    done
else
    echo "⚠ Docker not found. GROBID service requires Docker."
fi

# Download PDFFigures2 if not present
if [ ! -f "/usr/local/bin/pdffigures2.jar" ]; then
    echo "Downloading PDFFigures2..."
    wget https://github.com/allenai/pdffigures2/releases/download/v0.1.0/pdffigures2-0.1.0.jar -O pdffigures2.jar
    echo "✓ PDFFigures2 downloaded. Move it to appropriate location or update config.py"
fi

# Create .env file if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Application Settings
APP_NAME=PDFExtractor
APP_VERSION=1.0.0
DEBUG=False
LOG_LEVEL=INFO

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# External Services
GROBID_URL=http://localhost:8070
PDFFIGURES2_PATH=./pdffigures2.jar
NOUGAT_MODEL_PATH=facebook/nougat-base

# Storage
PAPER_FOLDER=./paper
OUTPUT_FORMAT=json
KEEP_INTERMEDIATE_FILES=True

# Processing
MAX_WORKERS=4
EXTRACTION_TIMEOUT=300
OCR_LANGUAGE=eng
USE_GPU=False

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
EOL
    echo "✓ .env file created"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To start the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Place your PDF in the 'paper' folder"
echo "3. Run the FastAPI server: python -m uvicorn app.main:app --reload"
echo "4. Access the API at: http://localhost:8000/api/v1/docs"
echo ""
echo "For paper folder extraction:"
echo "curl -X POST http://localhost:8000/api/v1/extract-from-folder"
echo ""