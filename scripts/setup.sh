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
mkdir -p paper/json
mkdir -p paper/text/sections
mkdir -p paper/text/subsections
mkdir -p paper/code
mkdir -p paper/ocr_math
mkdir -p cache/huggingface
mkdir -p cache/torch
mkdir -p cache/easyocr
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



# Check Docker (for GROBID)
if command -v docker &> /dev/null; then
    echo "✓ Docker found: $(docker --version)"
    
    # Check if docker-compose is available
    if command -v docker-compose &> /dev/null; then
        echo "✓ Docker Compose found: $(docker-compose --version)"
    else
        echo "⚠ Docker Compose not found. Please install Docker Compose."
    fi
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        echo "✓ Docker daemon is running"
    else
        echo "⚠ Docker daemon is not running. Please start Docker."
    fi
else
    echo "⚠ Docker not found. GROBID service requires Docker."
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
API_PORT=8002
API_PREFIX=/api/v1

# External Services
GROBID_URL=http://localhost:8070
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

# Cloudinary Configuration
CLOUDINARY_URL=cloudinary://your_api_key:your_api_secret@your_cloud_name
STORE_LOCALLY=True

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600

# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_VHOST=/

# Backblaze B2 Configuration
B2_APPLICATION_KEY_ID=your_b2_key_id
B2_APPLICATION_KEY=your_b2_application_key
B2_BUCKET_NAME=your_bucket_name
EOL
    echo "✓ .env file created"
fi

# Test Docker build
echo "Testing Docker build..."
if command -v docker &> /dev/null && docker info &> /dev/null; then
    echo "Building Docker image..."
    docker build -t scholar-extractor .
    if [ $? -eq 0 ]; then
        echo "✓ Docker image built successfully"
    else
        echo "⚠ Docker build failed. Check the Dockerfile and requirements."
    fi
else
    echo "⚠ Skipping Docker build test (Docker not available)"
fi

# Test Docker Compose
echo "Testing Docker Compose..."
if command -v docker-compose &> /dev/null && docker info &> /dev/null; then
    echo "Testing Docker Compose configuration..."
    docker-compose config
    if [ $? -eq 0 ]; then
        echo "✓ Docker Compose configuration is valid"
    else
        echo "⚠ Docker Compose configuration has issues"
    fi
else
    echo "⚠ Skipping Docker Compose test (Docker Compose not available)"
fi

# Set proper permissions for Docker volumes
echo "Setting permissions for Docker volumes..."
chmod -R 755 paper/
chmod -R 755 cache/
chmod -R 755 logs/

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To start the application:"
echo ""
echo "Local Development:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Place your PDF in the 'paper' folder"
echo "3. Run the FastAPI server: python -m uvicorn app.main:app --reload"
echo "4. Access the API at: http://localhost:8002/api/v1/docs"
echo ""
echo "Docker Development:"
echo "1. Build and run with Docker Compose: docker-compose up --build"
echo "2. Access the API at: http://localhost:8002/api/v1/docs"
echo "3. Stop the service: docker-compose down"
echo ""
echo "For paper folder extraction:"
echo "curl -X POST http://localhost:8002/api/v1/extract-from-folder"
echo ""
echo "Docker Commands:"
echo "- Build image: docker build -t scholar-extractor ."
echo "- Run container: docker run -p 8002:8002 -v \$(pwd)/paper:/app/paper scholar-extractor"
echo "- View logs: docker-compose logs -f extractor"
echo ""