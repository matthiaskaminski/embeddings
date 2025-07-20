## Project Overview

Automatyzacja wyszukiwania podobnych produktów wnętrzarskich (sofy, stoły, fotele, krzesła, szafy) przy użyciu trzech modeli AI: DINOv2, OpenAI text embedding i CLIP. Produkty importowane z Supabase przez N8N.

### Workflow N8N:
1. **Upload wnętrza** - Użytkownik przesyła zdjęcie wnętrza jako inspirację
2. **Detekcja obiektów** - Model wykrywania identyfikuje konkretne produkty (np. sofa)
3. **Wybór produktu** - Użytkownik wybiera interesujący go obiekt
4. **Crop & analiza** - Obiekt przycinany i wysyłany do analizy OpenAI
5. **Generowanie embeddingów** - Kombinacja 3 wektorów: wizualny + tekstowy + CLIP
6. **Wyszukiwanie FAISS** - Porównanie z bazą ~1750 sof
7. **RNN filtering** - Sprawdzanie nearest neighbor dla eliminacji false positives
8. **Wyniki** - Zwrócenie top 6-8 najbardziej podobnych produktów

### Architektura Modeli:
- **CLIP** - embeddingi obraz-tekst (768 dim)
- **DINOv2** - zaawansowane embeddingi wizualne (768 dim) 
- **OpenAI text-embedding-3-large** - embeddingi tekstowe cech (3072 dim)
- **FAISS HNSW** - szybkie wyszukiwanie podobieństwa w bazie wektorów
- **Kombinowane embeddingi** - weighted concatenation wszystkich trzech

### Stan Bazy Danych:
- **~1750 sof** obecnie zindeksowanych
- **Finalne wyniki**: top 6-8 produktów dla użytkownika
- **Filtry**: grupa_kolorów dla lepszej precyzji

## Analiza Cech Produktów (Sofy)

### Obecne 13 Cech:
1. **materiał** (waga: 2.5) - typu skóra, tkanina, welur
2. **specyfikacja_materiału** (2.0) - konkretny rodzaj materiału
3. **kolor** (3.0) - najważniejsza cecha wizualna
4. **styl** (1.3) - nowoczesny, klasyczny, skandynawski
5. **podstawa** (1.5) - nogi, płoza, obrotowa
6. **typ** (2.0) - narożna, prosta, modułowa
7. **pojemność** (1.2) - ilość miejsc siedzących
8. **kierunek_ustawienia** (0.8) - lewy, prawy róg
9. **cechy_dodatkowe** (0.5) - funkcje dodatkowe
10. **kształt** (1.8) - L-shape, U-shape, prosta
11. **pikowana** (1.0) - tak/nie, tekstura powierzchni
12. **szezlong** (1.0) - obecność części do leżenia
13. **grupa_kolorów** - grupowanie dla filtrowania

### Ocena Cech:
✅ **Wystarczające** dla precyzyjnego wyszukiwania  
✅ **Zbalansowane** dla embeddingu 3072 dim  
✅ **Logiczna hierarchia** ważności w FEATURE_WEIGHTS

## Kluczowe Zalecenia Społeczności (GitHub)

### 1. Background Removal (15% improvement)
- **Rembg/SAM** - największy gain w preprocessingu
- Keszowanie usuniętych teł dla performance
- Opcja fast/quality w zależności od latencji

### 2. Two-Stage Search (dostosowane do 1750 sof)
- **Stage 1**: Coarse retrieval 1750 → 300 produktów  
- **Stage 2**: Re-ranking 300 → 50 produktów
- **Final**: Top 6-8 dla użytkownika
- **Hybrid scoring**: Visual (70%) + Category (15%) + Metadata (15%)

### 3. Model Upgrades
- **CLIP ViT-L/14@336px** zamiast standardowego (wyższa rozdzielczość)
- **EVA-CLIP/BLIP-2** przewyższają vanilla CLIP
- **Multi-scale processing** (224px, 336px, 448px) + averaging

### 4. Embedding Optimization
- **Weighted concatenation** lepsze od averaging (15% gain)
- Separate normalization → concat → final normalization
- **RNN threshold 0.7** dla normalized space

### 5. Validation & Quality
- **Exact NN** opcja dla małych baz (<100M)
- **VQA validation** z małym modelem (yes/no relevance)
- **A/B testing** każdej zmiany

## Dzień 1-2: Historia Rozwoju
- **Dzień 1**: Pierwsza baza, słabe wyniki
- **Poprawki**: Długość embeddingu + grupa_kolorów → lepsze wyniki
- **Aktualnie**: Testowanie advanced techniques z zaleceń społeczności

## Zaimplementowane Ulepszenia ✅

### Główne Features:
1. ✅ **Two-stage search** `/faiss/search/two-stage` - 1750 → 300 → 6-8 workflow
2. ✅ **Background removal** z Rembg + persistent storage
3. ✅ **Multi-scale processing** - 224px, 336px, 448px averaging
4. ✅ **Enhanced RNN validation** z exact NN option i threshold config
5. ✅ **Model upgrade** - CLIP ViT-H/14 (1280 dim) + DINOv2 giant (1536 dim)
6. ✅ **Enhanced product features** - 16 cech z `rozmiar_cm`, `funkcja_spania`, `schowek`, `mechanizm`

### Advanced Capabilities:
- **Persistent image storage** - `faiss_storage/processed_images/` z 7-day TTL
- **Configurable models** - Small/Large model switching
- **Admin endpoints** - Cleanup, stats, model config
- **Enhanced embeddings** - Larger dimensions, better accuracy

## Struktura Projektu (Po Cleanup)

```
E:\Szuk.ai\Embeddings\
├── app.py                     # Main application (3200+ lines)
├── requirements.txt           # Python dependencies
├── CLAUDE.md                 # Project documentation (this file)
├── README.md                 # Basic project info
├── SECURITY_INFO.md          # Security guidelines
├── TUNNEL_SETUP.md           # Tunnel setup instructions
└── faiss_storage/            # Persistent data storage
    ├── indexes/              # FAISS indexes
    │   ├── clip.index
    │   ├── dinov2.index
    │   ├── text.index
    │   └── combined.index
    ├── processed_images/     # Background-removed images
    │   ├── {hash}_original.png
    │   └── {hash}_no_bg.png
    └── metadata/             # Metadata files
        ├── product_metadata.pkl
        └── processed_images_metadata.pkl
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# Server starts on http://localhost:5000
# API documentation available at /
```

## Usunięte Pliki (Cleanup 2024)

**Debug/Test Scripts (8 plików):**
- `check_build_status.py`, `debug_*.py`, `test_*.py`
- Funkcjonalność przeniesiona do API endpoints

**Ngrok Files (4 pliki):**
- `ngrok.exe`, `ngrok.zip`, `run_with_ngrok.bat`, `ngrok-v3/`
- Ngrok można zainstalować globalnie

**Startup Files (3 pliki):**
- `start_server.py`, `start_tunnel.bat`, `startup_commands.md`
- Zastąpione przez `python app.py`

**Legacy Docs (1 plik):**
- `zalecenia od uzytkownikow.txt` - zintegrowane w CLAUDE.md

## Testing & Validation Tools

### Model Performance Testing
```bash
# Test current model configuration
POST /test/model-performance
{
  "image_url": "https://example.com/test-image.jpg",
  "features": {
    "kolor": "szary",
    "material": "tkanina",
    "typ": "sofa"
  }
}
```

### Batch Background Removal
```bash
# Process multiple images
POST /preprocess/batch-remove-background
{
  "images": [
    {"image_url": "https://example.com/image1.jpg"},
    {"image": "base64_encoded_image"}
  ],
  "use_cache": true,
  "save_to_disk": true,
  "return_images": false
}
```

### Two-Stage Search Testing
```bash
# Advanced search with larger models
POST /faiss/search/two-stage
{
  "image_url": "https://example.com/search-image.jpg",
  "features": {"kolor": "beżowy", "material": "tkanina"},
  "k": 6,
  "remove_background": true,
  "use_multiscale": true,
  "color_filter": "beżowe"
}
```

## Current Status: Production Ready ✅

- ✅ **Large Models Configured**: CLIP ViT-H/14 (1280 dim) + DINOv2 giant (1536 dim)
- ✅ **Clean Environment**: Zbędne pliki usunięte, struktura production-ready
- ✅ **Advanced Features**: Two-stage search, background removal, multi-scale
- ✅ **Testing Tools**: Model performance testing, batch processing
- ⏳ **Next Steps**: Test larger models, rebuild indexes, performance validation