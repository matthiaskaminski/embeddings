# RunPod Template Setup Instructions

## Creating Szuk.AI Embeddings Template

### 1. Launch Base Pod
- **GPU**: RTX 4090 or better
- **Image**: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Container Disk**: 50GB minimum
- **Volume**: Optional (for persistent storage between different pods)

### 2. Template Preparation Commands

```bash
# Update system
apt update && apt upgrade -y

# Install system dependencies
apt install -y git git-lfs curl wget htop nvtop

# Initialize Git LFS globally
git lfs install --system

# Install Python packages that take time to compile
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.0
pip install faiss-gpu==1.7.4
pip install flask==2.3.3
pip install numpy==1.24.3
pip install pillow==10.0.1
pip install scikit-learn==1.3.0
pip install openai==1.3.0
pip install rembg==2.0.50

# Install Claude AI CLI
curl -sSL https://claude.ai/cli/install.sh | bash

# Pre-download some models to speed up first run
python -c "
import torch
import clip
# Pre-load CLIP models
clip.load('ViT-L/14', device='cpu')
clip.load('ViT-L/14@336px', device='cpu')
print('Models cached successfully')
"

# Create workspace directory
mkdir -p /workspace
cd /workspace

# Set up environment variables
echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Test GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Save as Template
1. Go to RunPod dashboard
2. Click "Save Template" on your pod
3. Name: "Szuk.AI-Embeddings-Ready"
4. Description: "PyTorch 2.0 + FAISS + Claude AI CLI for Szuk.AI embeddings"

### 4. Template Environment Variables (Optional)
```bash
PYTHONPATH=/workspace:/workspace/szuk-ai-embeddings
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8
```

## Using the Template

### Quick Deploy
1. Start pod with "Szuk.AI-Embeddings-Ready" template
2. Connect via SSH or web terminal
3. Run deployment:
```bash
curl -sSL https://raw.githubusercontent.com/[username]/szuk-ai-embeddings/main/setup.sh | bash
```

### Manual Deploy
```bash
cd /workspace
git clone https://github.com/[username]/szuk-ai-embeddings.git
cd szuk-ai-embeddings
git lfs pull
pip install -r requirements.txt
python app.py
```

## Resource Requirements

### Minimum
- **GPU**: RTX 3060 (12GB VRAM)
- **RAM**: 16GB
- **Storage**: 25GB

### Recommended  
- **GPU**: RTX 4090 (24GB VRAM)
- **RAM**: 32GB
- **Storage**: 50GB

### Performance Comparison
- **RTX 3060**: ~3-5 seconds per embedding
- **RTX 4090**: ~1-2 seconds per embedding  
- **A100**: ~0.5-1 second per embedding

## Cost Optimization

### Development
- Use RTX 4090: ~$0.50-0.70/hour
- Stop when not in use: $0/hour

### Production
- Use A100 for fastest processing
- Process in batches
- Auto-stop after completion

## Troubleshooting

### Common Issues
1. **Out of memory**: Use smaller batch sizes
2. **Git LFS timeout**: Increase timeout: `git config lfs.activitytimeout 300`
3. **CUDA errors**: Restart pod and check drivers