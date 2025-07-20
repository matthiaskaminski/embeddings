# Szuk.AI Embeddings - Furniture Similarity Search

Advanced AI-powered furniture similarity search system using CLIP, DINOv2, and OpenAI embeddings with FAISS indexing.

## 🎯 Features

- **Multi-modal embeddings**: CLIP (visual) + DINOv2 (features) + OpenAI (text)
- **Large model support**: ViT-L/14@336px + DINOv2 giant
- **Concatenation strategy**: 5376-dimensional combined embeddings  
- **Two-stage search**: Coarse retrieval → re-ranking → top results
- **Background removal**: Automated with Rembg
- **Async processing**: Background tasks for large datasets
- **RunPod ready**: One-click deployment on GPU cloud
- **Git LFS**: Efficient storage for FAISS indexes

## 🚀 Quick Deploy (RunPod)

```bash
curl -sSL https://raw.githubusercontent.com/[username]/szuk-ai-embeddings/main/setup.sh | bash
```

## 📊 Architecture

### Models Used
- **CLIP ViT-L/14@336px**: 768-dim visual embeddings
- **DINOv2 Giant**: 1536-dim advanced visual features  
- **OpenAI text-embedding-3-large**: 3072-dim text features
- **Combined**: 5376-dim concatenated embeddings

### Workflow
1. **Image preprocessing** → Background removal
2. **Multi-scale processing** → 224px, 336px, 448px
3. **Feature extraction** → Visual + text embeddings
4. **FAISS indexing** → Fast similarity search
5. **Two-stage search** → 1750 → 300 → 6-8 results

## 🔧 API Endpoints

### Build & Management
```bash
# Build new indexes
POST /faiss/build-async
{
  "products": [
    {
      "id": 1,
      "image_url": "https://...",
      "features": {
        "kolor": "szary", 
        "material": "tkanina",
        "typ": "sofa"
      }
    }
  ]
}

# Add products to existing indexes  
POST /faiss/add-async
{
  "products": [...],
  "append": true
}
```

### Search
```bash
# Two-stage similarity search
POST /faiss/search/two-stage
{
  "image_url": "https://...",
  "features": {
    "kolor": "beżowy",
    "material": "tkanina"
  },
  "k": 6,
  "remove_background": true,
  "use_multiscale": true
}
```

### Monitoring
```bash
# System health
GET /health

# Index statistics
GET /faiss/stats

# Task status
GET /task/{task_id}
```

## 🏗️ Local Development

### Prerequisites
- Python 3.10+
- CUDA 11.8+ 
- 12GB+ GPU memory
- Git LFS

### Setup
```bash
git clone https://github.com/[username]/szuk-ai-embeddings.git
cd szuk-ai-embeddings
git lfs pull
pip install -r requirements.txt
python app.py
```

## 🌐 Production Deployment

### RunPod (Recommended)
1. **Create template** from `runpod-template.md`
2. **Deploy**: `curl setup.sh | bash`
3. **Access**: ngrok tunnel or public IP
4. **Scale**: Start/stop on demand

### Docker
```bash
docker-compose up -d
```

### Costs
- **Development**: ~$0.60/hour (RTX 4090)
- **Production**: On-demand processing only
- **Storage**: Free with Git LFS (under 2GB)

## 📈 Performance

### Processing Speed
- **RTX 3060**: 3-5 sec/embedding
- **RTX 4090**: 1-2 sec/embedding  
- **A100**: 0.5-1 sec/embedding

### Search Performance
- **Index size**: 1750 products
- **Search time**: <100ms
- **Memory usage**: 5GB GPU + 8GB RAM

## 🔐 Security

### Authentication
All endpoints require:
```bash
X-API-Key: szuk_ai_embeddings_2024_secure_key
```

### Rate Limiting
- 100 requests/minute per IP
- Configurable limits

## 🛠️ Model Configuration

### Switch Models
```bash
# Large models (production)
POST /admin/switch-model-size
{"model_size": "large"}

# Small models (development)  
POST /admin/switch-model-size
{"model_size": "small"}
```

### Current Configuration
```bash
GET /admin/model-config
```

## 📚 Documentation

- **[Deployment Guide](DEPLOYMENT.md)** - Complete setup instructions
- **[RunPod Template](runpod-template.md)** - Template creation guide
- **[Project Details](CLAUDE.md)** - Technical specifications
- **[Security Info](SECURITY_INFO.md)** - Security guidelines

## 🐛 Troubleshooting

### Common Issues
- **Memory errors**: Use smaller batch sizes or small models
- **CUDA errors**: Restart pod and check GPU status
- **LFS timeouts**: Increase git timeout settings

### Support
```bash
# Check logs
tail -f logs/app.log

# Monitor GPU
nvidia-smi

# Test health
curl http://localhost:5000/health
```

## 📊 Data Pipeline

### Input → Processing → Output
```
Product Image → Background Removal → Multi-scale Processing
     ↓                    ↓                     ↓
CLIP Embedding ← DINOv2 Embedding ← Text Embedding  
     ↓                    ↓                     ↓
          Concatenation (5376 dims)
                    ↓
              FAISS Indexing
                    ↓
            Similarity Search
                    ↓
             Top 6-8 Results
```

## 🎯 Use Cases

- **E-commerce**: Product recommendation
- **Interior Design**: Style matching
- **Inventory**: Similar product search
- **Content**: Visual similarity detection

## 🚀 Future Enhancements

- [ ] Vector database integration (Pinecone, Weaviate)
- [ ] Real-time reindexing
- [ ] Multi-language support
- [ ] Advanced filtering options
- [ ] A/B testing framework

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch  
5. Create Pull Request

---

**Powered by**: PyTorch • CLIP • DINOv2 • FAISS • Flask • RunPod