# RunPod + Claude Code - Kompletny Przewodnik Setup

## 🌍 **Pytanie o Regiony:**

### **ZALECENIE: Wybierz Europę!**
- ✅ **Europa** - Mniejsze opóźnienia dla Twoich requestów z Polski
- ✅ **Stabilne połączenie** - Bliższa lokalizacja = mniej problemów z siecią
- ✅ **GDPR compliance** - Europejskie przepisy o danych

### **Regionów do wyboru:**
- 🇳🇱 **EU-West-1 (Holandia)** - NAJLEPSZY dla Polski
- 🇨🇿 **EU-Central-1 (Czechy)** - Bardzo dobry
- 🇺🇸 US-East/West - Tylko jeśli EU niedostępne

## 📋 **KROK PO KROKU - Pierwszy Setup**

### **ETAP 1: Stworzenie Poda (15 min)**

1. **Wejdź na [runpod.io](https://runpod.io)**
2. **Kliknij "Rent GPUs"**
3. **Wybierz GPU:**
   - **RTX 4090** (24GB) - ZALECANE dla development
   - **RTX 3080** (10GB) - Budget option
   - **A100** (40GB) - Jeśli potrzebujesz najwyższej wydajności

4. **Template Selection:**
   - **Wybierz:** `RunPod PyTorch 2.0.1`
   - **WAŻNE:** NIE wybieraj jeszcze custom template!

5. **Container Configuration:**
   ```
   Container Disk Space: 50GB
   Volume Disk Space: 20GB (opcjonalnie)
   Expose HTTP Ports: 5000
   ```

6. **Region Selection:**
   - **Wybierz:** EU-West-1 (Netherlands)

7. **Kliknij "Deploy"**

### **ETAP 2: Połączenie z Podem (5 min)**

1. **Poczekaj aż Status = "Running"**
2. **Kliknij "Connect"**
3. **Wybierz "Start Web Terminal"** lub "SSH"

### **ETAP 3: Instalacja Claude Code (10 min)**

```bash
# 1. Update systemu
apt update && apt upgrade -y

# 2. Zainstaluj curl jeśli nie ma
apt install -y curl

# 3. Zainstaluj Claude AI CLI
curl -sSL https://claude.ai/cli/install.sh | bash

# 4. Restart shell lub dodaj do PATH
source ~/.bashrc
# LUB
export PATH="$HOME/.local/bin:$PATH"

# 5. Sprawdź czy działa
claude --version

# 6. Login do Claude (jeśli potrzebne)
claude auth login
```

### **ETAP 4: Auto-Deploy Embeddings (3 min)**

```bash
# Jedną komendą uruchom cały system
curl -sSL https://raw.githubusercontent.com/matthiaskaminski/embeddings/main/setup.sh | bash
```

**Co się dzieje:**
- ✅ Instaluje Git LFS
- ✅ Klonuje Twoje repo
- ✅ Pobiera indexy FAISS (Git LFS pull)
- ✅ Instaluje Python dependencies  
- ✅ Uruchamia serwer na porcie 5000

### **ETAP 5: Sprawdzenie czy działa (2 min)**

```bash
# Test health endpoint
curl http://localhost:5000/health

# Sprawdź status indexów
curl http://localhost:5000/faiss/stats

# Sprawdź dostępne endpointy
curl http://localhost:5000/
```

### **ETAP 6: Setup Tunnelu dla N8N (5 min)**

#### **Opcja A: ngrok (Zalecane)**
```bash
# Zainstaluj ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
apt update && apt install ngrok

# Setup authtoken (pobierz z ngrok.com)
ngrok authtoken YOUR_TOKEN_HERE

# Utwórz tunnel
ngrok http 5000
```

#### **Opcja B: RunPod Public IP**
```bash
# Sprawdź external IP poda
curl -4 ifconfig.me

# URL będzie: http://[PUBLIC_IP]:5000
```

### **ETAP 7: Zapisanie jako Template (5 min)**

1. **W RunPod dashboard kliknij swój pod**
2. **Kliknij "Save as Template"**
3. **Nazwa:** `Szuk-AI-Embeddings-Claude-Ready`
4. **Opis:** `PyTorch + FAISS + Claude Code + Git LFS ready`
5. **Kliknij "Save Template"**

## 🚀 **NASTĘPNE DEPLOYMENTS (2 minuty!)**

Po stworzeniu template, kolejne deploymenty:

1. **"Deploy from Template"**
2. **Wybierz:** `Szuk-AI-Embeddings-Claude-Ready`
3. **Po uruchomieniu:**
   ```bash
   cd /workspace
   curl -sSL https://raw.githubusercontent.com/matthiaskaminski/embeddings/main/setup.sh | bash
   ```

## 📝 **Instrukcje dla Claude Code w Terminalu**

### **Prompt dla Claude Code na RunPod:**

```
Cześć! Jestem na RunPod GPU instance i potrzebuję uruchomić system Szuk.AI Embeddings.

KONTEKST:
- To system wyszukiwania podobieństw produktów meblowych
- Używa CLIP + DINOv2 + OpenAI embeddings z FAISS indexami
- Repo: https://github.com/matthiaskaminski/embeddings
- Główny plik: app.py (Flask API server)

OBECNY STAN:
- Jestem w terminalu RunPod RTX 4090
- System: Ubuntu 22.04 z PyTorch 2.0.1
- GPU: Dostępne i działające

CO CHCĘ ZROBIĆ:
1. Uruchomić auto-deployment script z GitHub
2. Sprawdzić czy serwer działa poprawnie
3. Setup ngrok tunnel dla external access
4. Test podstawowych API endpoints

KOMENDY DO WYKONANIA:
```bash
# Auto-deploy
curl -sSL https://raw.githubusercontent.com/matthiaskaminski/embeddings/main/setup.sh | bash

# Sprawdź status
curl http://localhost:5000/health

# Setup ngrok (jeśli potrzebne)
ngrok http 5000
```

Proszę pomóż mi wykonać te kroki i troubleshoot jeśli coś nie zadziała. Szczególnie zwróć uwagę na:
- GPU memory usage (nvidia-smi)
- Python dependencies conflicts
- FAISS indexes loading
- API endpoints functionality

Gotowy do startu!
```

## ⚡ **Pro Tips:**

### **Oszczędzanie Kosztów:**
```bash
# Zatrzymaj pod po pracy
# W RunPod dashboard: "Stop Pod"
# Koszt: $0/hour when stopped

# Restart z template
# Deploy from template → auto-setup w 2 minuty
```

### **Monitoring GPU:**
```bash
# Sprawdź GPU usage
nvidia-smi

# Monitor w czasie rzeczywistym
watch -n 1 nvidia-smi
```

### **Troubleshooting:**
```bash
# Sprawdź logi aplikacji
tail -f /workspace/embeddings/logs/app.log

# Sprawdź czy port 5000 działa
netstat -tulpn | grep 5000

# Test memory
free -h
```

### **Quick Commands:**
```bash
# Restart serwera
cd /workspace/embeddings && python app.py

# Update repo
cd /workspace/embeddings && git pull && git lfs pull

# Check system status
cd /workspace/embeddings && curl http://localhost:5000/health
```

## 🎯 **Finalny URL dla N8N:**

Po uruchomieniu ngrok będziesz mieć URL typu:
```
https://[random-id].ngrok.io
```

**Endpointy dla N8N:**
- `POST https://[random-id].ngrok.io/faiss/build-async`
- `POST https://[random-id].ngrok.io/faiss/add-async`
- `POST https://[random-id].ngrok.io/faiss/search/two-stage`

**Headers:**
```
X-API-Key: szuk_ai_embeddings_2024_secure_key
Content-Type: application/json
```

Ready to go! 🚀