# 🔒 ZABEZPIECZENIA SERWERA

## ✅ Dodane zabezpieczenia:

### 1. **API Key Authentication**
- **Klucz API**: `szuk_ai_embeddings_2024_secure_key`
- **Wymagany**: dla wszystkich endpointów AI (CLIP, DINOv2, Text, etc.)
- **Header**: `X-API-Key: szuk_ai_embeddings_2024_secure_key`

### 2. **Rate Limiting**
- **Limit**: 100 requestów na minutę na IP
- **Automatyczne czyszczenie**: stare wpisy usuwane co minutę
- **Response**: HTTP 429 przy przekroczeniu limitu

### 3. **Publiczne endpointy** (bez API key, tylko rate limiting):
- `GET /` - Dokumentacja
- `GET /health` - Status serwera
- `GET /faiss/stats` - Statystyki FAISS

### 4. **Chronione endpointy** (wymagają API key):
- `POST /clip` - CLIP embeddings
- `POST /dino` - DINOv2 embeddings
- `POST /text` - Text embeddings
- `POST /combined` - Kombinowane embeddings
- `POST /faiss/build` - Budowanie indeksu
- `POST /faiss/search` - Wyszukiwanie
- `POST /test/rnn` - Test RNN
- `POST /batch` - Batch processing
- `POST /debug/verify-norm` - Weryfikacja norm

---

## 🔧 UŻYCIE W n8n:

### **HTTP Request Node Configuration:**

#### **Headers** (wymagane dla chronionych endpointów):
```json
{
  "Content-Type": "application/json",
  "X-API-Key": "szuk_ai_embeddings_2024_secure_key"
}
```

#### **Przykład request kombinowanych embeddingów:**
```bash
curl -X POST YOUR_TUNNEL_URL/combined \
  -H "Content-Type: application/json" \
  -H "X-API-Key: szuk_ai_embeddings_2024_secure_key" \
  -d '{
    "image": "base64_encoded_image",
    "features": {
      "material": "bawełna",
      "kolor": "beżowy"
    }
  }'
```

#### **Przykład request wyszukiwania:**
```bash
curl -X POST YOUR_TUNNEL_URL/faiss/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: szuk_ai_embeddings_2024_secure_key" \
  -d '{
    "embed_type": "combined",
    "image": "base64_data",
    "features": {...},
    "k": 10
  }'
```

#### **Test bez API key (powinien zwrócić 401):**
```bash
curl -X POST YOUR_TUNNEL_URL/clip \
  -H "Content-Type: application/json" \
  -d '{"image": "test"}'

# Response: {"error": "Invalid or missing API key"}
```

---

## 🛡️ DODATKOWE ZABEZPIECZENIA:

### **Firewall Windows**:
- ✅ Wyłączony dla testów
- ⚠️ **Uwaga**: Serwer jest dostępny z internetu przez tunnel

### **IP Whitelisting** (opcjonalne):
Jeśli chcesz ograniczyć dostęp tylko do określonych IP, możesz dodać:
```python
ALLOWED_IPS = ['YOUR_N8N_IP', '192.168.0.0/24']  # Twoja sieć lokalna
```

### **HTTPS** (opcjonalne):
Dla produkcji warto dodać SSL/TLS certyfikaty.

---

## 🚀 GOTOWY SERWER:

**Status**: ✅ Zabezpieczony i gotowy do użycia
**URL lokalny**: http://192.168.0.128:5000
**API Key**: `szuk_ai_embeddings_2024_secure_key`
**Rate limit**: 100 req/min per IP

### **Teraz możesz bezpiecznie:**
1. Uruchomić tunnel (ngrok/localtunnel)
2. Używać publicznego URL w n8n
3. Testować z automatyzacją

**Wszyscy nieautoryzowani użytkownicy otrzymają błąd 401! 🔒**