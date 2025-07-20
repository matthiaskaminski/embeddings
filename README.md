# Flask Embedding Server - Furniture Similarity Search

Zaawansowany serwer Flask do wyszukiwania podobnych mebli z wykorzystaniem multi-modal embeddingów.

## Funkcje

- **Multi-modal embeddings**: CLIP + DINOv2 + OpenAI Text
- **FAISS Vector Database**: Szybkie wyszukiwanie podobieństwa
- **GPU Acceleration**: Automatyczna optymalizacja dla RTX 3060
- **Reciprocal Nearest Neighbors**: Test wzajemności podobieństwa
- **Batch Processing**: Przetwarzanie wielu produktów naraz

## Instalacja

1. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

2. Uruchom serwer:
```bash
python app.py
```

## Konfiguracja ngrok

1. Skonfiguruj ngrok z tokenem:
```bash
./ngrok.exe authtoken 2zoy2kjtRzuXJfmQ3psR6C7iTFU_2ihkL4qpC9tMgmDVg8DTn
```

2. Uruchom tunnel:
```bash
./ngrok.exe http 5000
```

## Endpointy

### POST /clip
Generuje embeddingi CLIP z obrazu.

### POST /dino
Generuje embeddingi DINOv2 z obrazu.

### POST /text
Generuje embeddingi tekstowe z cech produktu.

**Przykład żądania:**
```json
{
  "features": {
    "material": "bawełna",
    "kolor": "oliwkowy zielony",
    "styl": "nowoczesna",
    "typ": "narożna"
  }
}
```

### POST /combined
Generuje kombinowane embeddingi (CLIP + DINOv2 + Text).

**Przykład żądania:**
```json
{
  "image": "base64_encoded_image",
  "features": {
    "material": "bawełna",
    "kolor": "oliwkowy zielony"
  },
  "weights": {
    "clip": 0.4,
    "dinov2": 0.3,
    "text": 0.3
  }
}
```

### POST /faiss/build
Buduje indeks FAISS z produktów.

**Przykład żądania:**
```json
{
  "products": [
    {
      "id": 1,
      "image": "base64_data",
      "features": {
        "material": "bawełna",
        "kolor": "beżowy"
      }
    }
  ],
  "index_type": "hnsw"
}
```

### POST /faiss/search
Wyszukuje podobne produkty.

**Przykład żądania:**
```json
{
  "embed_type": "combined",
  "image": "base64_data",
  "features": {
    "material": "bawełna"
  },
  "k": 10
}
```

### POST /test/rnn
Testuje Reciprocal Nearest Neighbors.

**Przykład żądania:**
```json
{
  "embed_type": "combined",
  "k": 5,
  "test_samples": 50
}
```

### POST /batch
Przetwarzanie batch.

**Przykład żądania:**
```json
{
  "products": [
    {
      "id": 1,
      "image": "base64_data",
      "features": {...}
    }
  ],
  "embed_types": ["combined"]
}
```

## Cechy produktów (Sofa)

System obsługuje następujące cechy sof:
- `material`: materiał (np. "bawełna", "poliester")
- `specyfikacja_materialu`: szczegóły materiału (np. "sztruks")
- `kolor`: kolor (np. "oliwkowy zielony")
- `styl`: styl (np. "nowoczesna", "skandynawska")
- `typ`: typ (np. "narożna", "prosta")
- `pojemnosc`: pojemność (np. "3-osobowa")
- `ksztalt`: kształt (np. "prostokątna")
- `podstawa`: podstawa (np. "małe podnóżki")
- `kierunek_ustawienia`: kierunek (np. "lewostronna")
- `szezlong`: szezlong (np. "z szezlongiem")
- `pikowana`: pikowanie (np. "pikowana")
- `cechy_dodatkowe`: dodatkowe cechy

## Użycie z n8n

1. Uruchom serwer lokalnie
2. Skonfiguruj ngrok i skopiuj publiczny URL
3. W n8n używaj HTTP Request nodes z różnymi endpointami

## Specyfikacja techniczna

- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **Modele**: CLIP ViT-L/14, DINOv2 ViT-L/14, text-embedding-3-large
- **Wagi domyślne**: CLIP (40%), DINOv2 (30%), Text (30%)
- **FAISS**: IndexHNSW dla optymalnej wydajności
- **Pojemność**: Do 10K produktów na kategorię