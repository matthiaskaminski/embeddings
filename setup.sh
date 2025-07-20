#!/bin/bash

# Szuk.AI Embeddings - RunPod Auto-Deployment Script
# Usage: curl -sSL https://raw.githubusercontent.com/[username]/szuk-ai-embeddings/main/setup.sh | bash

set -e  # Exit on any error

echo "🚀 Starting Szuk.AI Embeddings deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in RunPod environment
if [ -n "$RUNPOD_POD_ID" ]; then
    print_status "RunPod environment detected (Pod ID: $RUNPOD_POD_ID)"
    INSTALL_DIR="/workspace/szuk-ai-embeddings"
else
    print_status "Local environment detected"
    INSTALL_DIR="./szuk-ai-embeddings"
fi

# Install system dependencies if needed
print_status "Checking system dependencies..."
if ! command -v git &> /dev/null; then
    print_status "Installing git..."
    apt update && apt install -y git
fi

if ! command -v curl &> /dev/null; then
    print_status "Installing curl..."
    apt update && apt install -y curl
fi

# Install Git LFS if not available
if ! command -v git-lfs &> /dev/null; then
    print_status "Installing Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt install -y git-lfs
fi

# Initialize Git LFS
print_status "Initializing Git LFS..."
git lfs install

# Clone the repository
print_status "Cloning Szuk.AI Embeddings repository..."
if [ -d "$INSTALL_DIR" ]; then
    print_warning "Directory $INSTALL_DIR already exists. Removing..."
    rm -rf "$INSTALL_DIR"
fi

# Replace with actual repository URL
REPO_URL="https://github.com/[username]/szuk-ai-embeddings.git"
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Pull LFS files
print_status "Downloading FAISS indexes and metadata..."
git lfs pull

# Check Python version
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed!"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $PYTHON_VERSION"

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Install Claude AI CLI if not present
if ! command -v claude &> /dev/null; then
    print_status "Installing Claude AI CLI..."
    curl -sSL https://claude.ai/cli/install.sh | bash
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check GPU availability
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU detected: $GPU_INFO"
else
    print_warning "No GPU detected. Running in CPU mode."
fi

# Create logs directory
mkdir -p logs

# Set environment variables
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"

# Check if FAISS indexes exist
if [ -d "faiss_storage/indexes" ] && [ "$(ls -A faiss_storage/indexes)" ]; then
    print_success "FAISS indexes found and loaded"
    INDEX_COUNT=$(ls faiss_storage/indexes/*.index 2>/dev/null | wc -l)
    print_status "Found $INDEX_COUNT FAISS index files"
else
    print_warning "No FAISS indexes found. Will need to build from scratch."
fi

# Start the application
print_status "Starting Szuk.AI Embeddings server..."
print_success "🎉 Deployment completed successfully!"
print_status "Repository location: $INSTALL_DIR"
print_status "To start the server manually: cd $INSTALL_DIR && python app.py"
print_status ""
print_status "Auto-starting server in 3 seconds..."
sleep 3

# Start the Flask application
cd "$INSTALL_DIR"
python app.py