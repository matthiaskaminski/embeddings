# Konfiguracja tunelu dla n8n

Twój serwer Flask działa na:
- **Lokalny URL**: http://localhost:5000
- **Sieć lokalna**: http://192.168.0.128:5000

## Opcje tunelowania:

### Opcja 1: Aktualizacja ngrok (Zalecana)

1. Pobierz najnowszy ngrok v3 z https://ngrok.com/download
2. Zastąp stary plik ngrok.exe
3. Uruchom:
```bash
./ngrok.exe config add-authtoken 2zoy2kjtRzuXJfmQ3psR6C7iTFU_2ihkL4qpC9tMgmDVg8DTn
./ngrok.exe http 5000
```

### Opcja 2: Alternatywne tunele

#### Cloudflare Tunnel (Bezpłatny)
```bash
# Zainstaluj cloudflared
# Uruchom tunnel
cloudflared tunnel --url http://localhost:5000
```

#### LocalTunnel (Bezpłatny) - AKTYWNY
```bash
# Zainstaluj przez npm
npm install -g localtunnel

# Uruchom tunnel ze stałym subdomain
lt --port 5000 --subdomain szukaiembeddings

# Stały URL: https://szukaiembeddings.loca.lt
```

#### Serveo (Bezpłatny, przez SSH)
```bash
ssh -R 80:localhost:5000 serveo.net
```

### Opcja 3: Użycie IP lokalnego w n8n

Jeśli n8n działa w tej samej sieci lokalnej:
- Użyj URL: `http://192.168.0.128:5000`

## Gotowe endpointy dla n8n:

Po uruchomieniu tunelu, zastąp `YOUR_TUNNEL_URL` rzeczywistym URL:

### 1. Test połączenia
```
GET YOUR_TUNNEL_URL/
```

### 2. Generowanie embeddingów CLIP
```
POST YOUR_TUNNEL_URL/clip
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

### 3. Generowanie embeddingów kombinowanych
```
POST YOUR_TUNNEL_URL/combined
Content-Type: application/json

{
  "image": "base64_encoded_image_data",
  "features": {
    "material": "bawełna",
    "kolor": "beżowy",
    "styl": "nowoczesna"
  }
}
```

### 4. Budowanie indeksu FAISS
```
POST YOUR_TUNNEL_URL/faiss/build
Content-Type: application/json

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
  ]
}
```

### 5. Wyszukiwanie podobnych produktów
```
POST YOUR_TUNNEL_URL/faiss/search
Content-Type: application/json

{
  "embed_type": "combined",
  "image": "base64_data",
  "features": {
    "material": "bawełna"
  },
  "k": 10
}
```

### 6. Test normalizacji L2
```
POST YOUR_TUNNEL_URL/debug/verify-norm
Content-Type: application/json

{
  "image": "base64_data",
  "features": {
    "material": "bawełna",
    "kolor": "beżowy"
  }
}
```

## Status serwera:
✅ Serwer Flask uruchomiony na porcie 5000
✅ GPU RTX 3060 aktywne (2.01 GB VRAM użyte)
✅ Wszystkie modele załadowane:
   - CLIP ViT-L/14
   - DINOv2 ViT-L/14 
   - text-embedding-3-large
✅ Kompletna normalizacja L2 zaimplementowana
✅ Wszystkie embeddingi mają L2 norm = 1.0

Wybierz jedną z opcji tunelowania i możesz zacząć testować z n8n!