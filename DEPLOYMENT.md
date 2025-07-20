# Szuk.AI Embeddings - Deployment Guide

## 🚀 Quick Start (RunPod)

### 1-Click Deployment
```bash
curl -sSL https://raw.githubusercontent.com/[username]/szuk-ai-embeddings/main/setup.sh | bash
```

## 📋 Prerequisites

### RunPod Account Setup
1. Create account at [runpod.io](https://runpod.io)
2. Add payment method
3. (Optional) Create persistent volume for data

### GitHub Repository Setup
1. Fork or clone this repository
2. Update setup.sh with your repository URL (line 43)
3. Ensure Git LFS is configured

## 🎯 Deployment Options

### Option A: Template-Based (Recommended)

#### Step 1: Create RunPod Template
1. Launch RTX 4090 pod with PyTorch image
2. Follow instructions in `runpod-template.md`
3. Save as template "Szuk.AI-Embeddings-Ready"

#### Step 2: Deploy from Template
1. Start pod with your template
2. Connect via SSH/terminal
3. Run: `curl -sSL [your-setup-url] | bash`
4. Server starts automatically on port 5000

### Option B: Manual Deployment

```bash
# Connect to RunPod instance
cd /workspace

# Clone repository
git clone https://github.com/[username]/szuk-ai-embeddings.git
cd szuk-ai-embeddings

# Download LFS files (indexes, metadata)
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Install Claude AI CLI
curl -sSL https://claude.ai/cli/install.sh | bash

# Start server
python app.py
```

### Option C: Docker Deployment

```bash
# Build image
docker build -t szuk-ai-embeddings .

# Run container
docker-compose up -d
```

## 🔧 Configuration

### Environment Variables
```bash
export PYTHONPATH=/workspace/szuk-ai-embeddings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

### GPU Requirements
- **Minimum**: RTX 3060 (12GB VRAM)
- **Recommended**: RTX 4090 (24GB VRAM)
- **Enterprise**: A100 (40GB VRAM)

### Memory Usage
- **Large models**: ~5GB GPU + 8GB RAM
- **Small models**: ~3GB GPU + 4GB RAM

## 🌐 Access & Tunneling

### Local Access
- Server runs on: `http://localhost:5000`
- Health check: `http://localhost:5000/health`

### External Access (for N8N)
Choose one option:

#### Option 1: ngrok (Recommended)
```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Setup authtoken (get from ngrok.com)
ngrok authtoken YOUR_TOKEN

# Create tunnel
ngrok http 5000
```

#### Option 2: RunPod Public IP
- Use RunPod's public IP (if available)
- Configure firewall rules for port 5000

#### Option 3: Cloudflare Tunnel
```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Create tunnel
cloudflared tunnel --url http://localhost:5000
```

## 📊 API Endpoints

### Core Endpoints
- `POST /faiss/build-async` - Build new indexes
- `POST /faiss/add-async` - Add products to indexes  
- `POST /faiss/search/two-stage` - Search similar products
- `GET /health` - System health check
- `GET /faiss/stats` - Index statistics

### Authentication
All endpoints require header:
```
X-API-Key: szuk_ai_embeddings_2024_secure_key
```

## 🔄 Data Management

### Backup Strategy
1. **Automatic**: Git commits save index changes
2. **Manual**: Push to GitHub after major updates
3. **Sync**: Pull latest indexes when starting new pod

### Index Synchronization
```bash
# Upload current indexes
git add faiss_storage/
git commit -m "Update indexes - $(date)"
git push origin main

# Download latest indexes (new pod)
git pull origin main
git lfs pull
```

## 🐛 Troubleshooting

### Common Issues

#### "Out of memory" errors
```bash
# Solution 1: Use smaller batch sizes
# Edit app.py: MAX_BATCH_SIZE = 10

# Solution 2: Switch to small models
curl -X POST -H "X-API-Key: szuk_ai_embeddings_2024_secure_key" \
  -H "Content-Type: application/json" \
  -d '{"model_size": "small"}' \
  http://localhost:5000/admin/switch-model-size
```

#### "Git LFS timeout" errors
```bash
git config lfs.activitytimeout 300
git config lfs.dialtimeout 30
git lfs pull
```

#### "CUDA out of memory"
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Restart application
pkill -f python
python app.py
```

#### "Port already in use"
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 [PID]
```

### Performance Optimization

#### For RTX 3060 (12GB)
- Use small models: `model_size: "small"`
- Batch size: 5-10 products
- Expected: 3-5 seconds per embedding

#### For RTX 4090 (24GB)  
- Use large models: `model_size: "large"`
- Batch size: 10-20 products
- Expected: 1-2 seconds per embedding

#### For A100 (40GB)
- Use large models with larger batches
- Batch size: 20-50 products  
- Expected: 0.5-1 second per embedding

## 💰 Cost Management

### Development Phase
- **Pod Type**: RTX 4090 (~$0.60/hour)
- **Usage**: 2-4 hours per session
- **Monthly**: ~$50-100

### Production Phase
- **Strategy**: Process batches then stop
- **Pod Type**: A100 for fastest processing
- **Usage**: On-demand only
- **Cost**: Pay only for processing time

### Cost Optimization Tips
1. **Stop pods** when not processing
2. **Use spot instances** for 50% savings
3. **Process in batches** to minimize startup time
4. **Monitor usage** with RunPod dashboard

## 🔐 Security

### API Key Management
- Change default API key in production
- Use environment variables for keys
- Rotate keys regularly

### Network Security
- Use HTTPS tunnels (ngrok, cloudflare)
- Restrict access to N8N IP only
- Monitor access logs

## 📞 Support

### Logs Location
- **Application**: `logs/app.log`
- **Startup**: `app_startup.log`
- **GPU**: `nvidia-smi` for monitoring

### Monitoring Commands
```bash
# Check GPU usage
nvidia-smi

# Check system resources
htop

# Check application logs
tail -f logs/app.log

# Check index statistics
curl -s http://localhost:5000/faiss/stats
```

### Getting Help
1. Check logs for error messages
2. Review this troubleshooting section
3. Test with smaller datasets first
4. Verify GPU memory availability