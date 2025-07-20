import base64
import io
import torch
import clip
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import warnings
import gc
import openai
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import json
import hashlib
import time
import os
import pickle
from functools import wraps
import secrets
import requests
from urllib.parse import urlparse
import threading
import uuid
from collections import defaultdict
try:
    from rembg import remove
    REMBG_AVAILABLE = True
    print("Rembg loaded successfully")
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: Rembg not available. Background removal disabled.")

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Security configuration
API_KEY = "szuk_ai_embeddings_2024_secure_key"  # Klucz API do autoryzacji
RATE_LIMIT = {}  # Rate limiting storage
MAX_REQUESTS_PER_MINUTE = 100  # Max 100 requests per minute per IP

# OpenAI API configuration
openai.api_key = "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"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

clip_model = None
clip_preprocess = None
dinov2_model = None
dinov2_processor = None

# FAISS indexes
faiss_indexes = {
    'clip': None,
    'dinov2': None,
    'text': None,
    'combined': None
}

# Product metadata storage
product_metadata = []
processed_images_metadata = {}  # Hash -> metadata mapping for processed images

# Storage paths for persistent data
STORAGE_DIR = "faiss_storage"
FAISS_INDEX_DIR = os.path.join(STORAGE_DIR, "indexes")
PROCESSED_IMAGES_DIR = os.path.join(STORAGE_DIR, "processed_images")
EMBEDDINGS_NO_BG_DIR = os.path.join(STORAGE_DIR, "embeddings_no_bg")
METADATA_DIR = os.path.join(STORAGE_DIR, "metadata")

METADATA_FILE = os.path.join(METADATA_DIR, "product_metadata.pkl")
PROCESSED_IMAGES_METADATA_FILE = os.path.join(METADATA_DIR, "processed_images_metadata.pkl")

# Ensure storage directories exist
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_NO_BG_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Background task tracking
background_tasks = {}
task_lock = threading.Lock()

# Default embedding weights
EMBEDDING_WEIGHTS = {
    'clip': 0.4,
    'dinov2': 0.3,
    'text': 0.3
}

# Enhanced embedding weights for better furniture matching
EMBEDDING_WEIGHTS_ENHANCED = {
    'clip': 0.45,      # More visual focus
    'dinov2': 0.40,    # Strong shape/structure focus
    'text': 0.15       # Reduced text weight
}

# Feature importance weights for text embedding
FEATURE_WEIGHTS = {
    'kolor': 3.0,           # Color is very important
    'material': 2.5,        # Material is crucial
    'specyfikacja_materialu': 2.0,
    'typ': 2.0,            # Type of sofa
    'ksztalt': 1.8,        # Shape matters
    'rozmiar_cm': 1.6,     # Actual dimensions in cm (new)
    'podstawa': 1.5,       # Base/legs are visible
    'styl': 1.3,           # Style matters but less
    'pojemnosc': 1.2,      # Size matters (seats count)
    'funkcja_spania': 1.1, # Sleep function (new, split from cechy_dodatkowe)
    'schowek': 1.0,        # Storage function (new, split from cechy_dodatkowe)
    'pikowana': 1.0,       # Additional features
    'szezlong': 1.0,
    'kierunek_ustawienia': 0.8,
    'mechanizm': 0.7,      # Reclining, adjustable (new)
    'cechy_dodatkowe': 0.5 # Remaining additional features
}

# Color groups for filtering
COLOR_GROUPS = {
    'beżowe': ['beżowy', 'jasny beżowy', 'ciemny beżowy', 'kremowy', 'kremowobiały', 'écru', 'piaskowy'],
    'białe': ['biały', 'kremowobiały', 'mleczny', 'śnieżnobiały', 'perłowy'],
    'szare': ['szary', 'jasny szary', 'ciemny szary', 'antracytowy', 'grafitowy', 'stalowy'],
    'czarne': ['czarny', 'antracytowy', 'grafitowy'],
    'brązowe': ['brązowy', 'jasny brązowy', 'ciemny brązowy', 'kasztanowy', 'orzechowy', 'taupe', 'ciemny taupe'],
    'zielone': ['zielony', 'jasny zielony', 'ciemny zielony', 'khaki', 'oliwkowy', 'butelkowy'],
    'niebieskie': ['niebieski', 'jasny niebieski', 'ciemny niebieski', 'granatowy', 'denim'],
    'czerwone': ['czerwony', 'bordowy', 'wiśniowy', 'karminowy'],
    'różowe': ['różowy', 'jasny różowy', 'pudrowy', 'fuksja'],
    'żółte': ['żółty', 'cytrynowy', 'musztardowy', 'złoty'],
    'pomarańczowe': ['pomarańczowy', 'terakota', 'rdza', 'miedziany'],
    'fioletowe': ['fioletowy', 'liliowy', 'lawendowy', 'śliwkowy']
}

# Background removal cache (in-memory for fast access)
background_removal_cache = {}
BG_CACHE_MAX_SIZE = 100  # Limit cache size
BG_CACHE_TTL = 3600  # 1 hour TTL

# Persistent storage configuration
PROCESSED_IMAGES_MAX_AGE = 7 * 24 * 3600  # 7 days
PROCESSED_IMAGES_CLEANUP_INTERVAL = 24 * 3600  # Daily cleanup

# Model configuration
MODEL_CONFIG = {
    'clip': {
        'small': {'name': 'ViT-L/14', 'dim': 768},
        'large': {'name': 'ViT-L/14@336px', 'dim': 768}
    },
    'dinov2': {
        'small': {'name': 'facebook/dinov2-large', 'dim': 1024},
        'large': {'name': 'facebook/dinov2-giant', 'dim': 1536}
    }
}

# Current model selection (configurable)
CURRENT_MODEL_SIZE = 'large'  # 'small' or 'large'

def get_color_group(color):
    """Get color group for a given color"""
    if not color or color == 'null':
        return None
    
    color_lower = color.lower().strip()
    for group, colors in COLOR_GROUPS.items():
        if any(c in color_lower for c in colors):
            return group
    return None

def load_processed_images_metadata():
    """Load processed images metadata from disk"""
    global processed_images_metadata
    try:
        if os.path.exists(PROCESSED_IMAGES_METADATA_FILE):
            with open(PROCESSED_IMAGES_METADATA_FILE, 'rb') as f:
                processed_images_metadata = pickle.load(f)
            print(f"Loaded {len(processed_images_metadata)} processed images metadata")
        else:
            processed_images_metadata = {}
    except Exception as e:
        print(f"Error loading processed images metadata: {e}")
        processed_images_metadata = {}

def save_processed_images_metadata():
    """Save processed images metadata to disk"""
    try:
        with open(PROCESSED_IMAGES_METADATA_FILE, 'wb') as f:
            pickle.dump(processed_images_metadata, f)
        print(f"Saved {len(processed_images_metadata)} processed images metadata")
        return True
    except Exception as e:
        print(f"Error saving processed images metadata: {e}")
        return False

def get_image_hash(image):
    """Generate hash for image content"""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    return hashlib.md5(image_bytes.getvalue()).hexdigest()

def save_processed_image(image_hash, original_image, processed_image):
    """Save both original and processed image to disk"""
    try:
        original_path = os.path.join(PROCESSED_IMAGES_DIR, f"{image_hash}_original.png")
        processed_path = os.path.join(PROCESSED_IMAGES_DIR, f"{image_hash}_no_bg.png")
        
        # Save images
        original_image.save(original_path, 'PNG')
        processed_image.save(processed_path, 'PNG')
        
        # Update metadata
        processed_images_metadata[image_hash] = {
            'original_path': original_path,
            'processed_path': processed_path,
            'timestamp': time.time(),
            'original_size': original_image.size,
            'processed_size': processed_image.size
        }
        
        # Save metadata
        save_processed_images_metadata()
        
        print(f"Saved processed image: {image_hash}")
        return True
        
    except Exception as e:
        print(f"Error saving processed image {image_hash}: {e}")
        return False

def load_processed_image(image_hash):
    """Load processed image from disk"""
    try:
        if image_hash not in processed_images_metadata:
            return None
            
        metadata = processed_images_metadata[image_hash]
        processed_path = metadata['processed_path']
        
        if os.path.exists(processed_path):
            return Image.open(processed_path)
        else:
            # Clean up missing file from metadata
            del processed_images_metadata[image_hash]
            save_processed_images_metadata()
            return None
            
    except Exception as e:
        print(f"Error loading processed image {image_hash}: {e}")
        return None

def cleanup_old_processed_images():
    """Remove old processed images based on age"""
    try:
        current_time = time.time()
        to_remove = []
        
        for image_hash, metadata in processed_images_metadata.items():
            age = current_time - metadata['timestamp']
            if age > PROCESSED_IMAGES_MAX_AGE:
                to_remove.append(image_hash)
        
        for image_hash in to_remove:
            metadata = processed_images_metadata[image_hash]
            
            # Remove files
            for path in [metadata['original_path'], metadata['processed_path']]:
                if os.path.exists(path):
                    os.remove(path)
            
            # Remove from metadata
            del processed_images_metadata[image_hash]
        
        if to_remove:
            save_processed_images_metadata()
            print(f"Cleaned up {len(to_remove)} old processed images")
        
        return len(to_remove)
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return 0

def filter_products_by_color_group(target_color_group):
    """Get indices of products that match the color group"""
    if not target_color_group:
        return list(range(len(product_metadata)))
    
    matching_indices = []
    for i, metadata in enumerate(product_metadata):
        product_color = metadata.get('features', {}).get('kolor', '')
        product_color_group = get_color_group(product_color)
        if product_color_group == target_color_group:
            matching_indices.append(i)
    
    return matching_indices

def get_client_ip():
    """Get client IP address"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr

def rate_limit_check():
    """Check rate limiting"""
    client_ip = get_client_ip()
    current_time = time.time()
    
    # Clean old entries (older than 1 minute)
    cutoff_time = current_time - 60
    RATE_LIMIT[client_ip] = [timestamp for timestamp in RATE_LIMIT.get(client_ip, []) if timestamp > cutoff_time]
    
    # Check if limit exceeded
    if len(RATE_LIMIT.get(client_ip, [])) >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    # Add current request
    if client_ip not in RATE_LIMIT:
        RATE_LIMIT[client_ip] = []
    RATE_LIMIT[client_ip].append(current_time)
    
    return True

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check rate limiting first
        if not rate_limit_check():
            return jsonify({'error': 'Rate limit exceeded. Max 100 requests per minute.'}), 429
        
        # Check API key
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key or api_key != API_KEY:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def public_endpoint(f):
    """Decorator for public endpoints (only rate limiting)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not rate_limit_check():
            return jsonify({'error': 'Rate limit exceeded. Max 100 requests per minute.'}), 429
        return f(*args, **kwargs)
    return decorated_function

def load_models():
    global clip_model, clip_preprocess, dinov2_model, dinov2_processor
    
    # Load processed images metadata
    print("Loading processed images metadata...")
    load_processed_images_metadata()
    
    # Get model configurations
    clip_config = MODEL_CONFIG['clip'][CURRENT_MODEL_SIZE]
    dinov2_config = MODEL_CONFIG['dinov2'][CURRENT_MODEL_SIZE]
    
    print(f"Loading CLIP model: {clip_config['name']} ({clip_config['dim']} dims)")
    clip_model, clip_preprocess = clip.load(clip_config['name'], device=device)
    clip_model.eval()
    
    if torch.cuda.is_available():
        print(f"CLIP model loaded on GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print(f"Loading DINOv2 model: {dinov2_config['name']} ({dinov2_config['dim']} dims)")
    dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_config['name'])
    dinov2_model = AutoModel.from_pretrained(dinov2_config['name']).to(device)
    dinov2_model.eval()
    
    if torch.cuda.is_available():
        print(f"Total GPU memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    
    print(f"Models loaded successfully! Using {CURRENT_MODEL_SIZE} models:")
    print(f"  - CLIP: {clip_config['name']} ({clip_config['dim']} dims)")
    print(f"  - DINOv2: {dinov2_config['name']} ({dinov2_config['dim']} dims)")
    
    # Load existing FAISS indexes and metadata
    print("Loading existing FAISS indexes...")
    load_faiss_indexes()

def features_to_text(features, use_weighted=True):
    """Convert product features JSON to natural language text with optional weighting"""
    try:
        # Handle null values (both None and "null" string)
        def safe_get(key, default=""):
            value = features.get(key, default)
            return value if value and value != "null" and value is not None else default
        
        if not use_weighted:
            # Original implementation for compatibility
            parts = []
            parts.append("Sofa")
            
            if safe_get("typ"):
                parts.append(safe_get("typ"))
            if safe_get("pojemnosc"):
                parts.append(safe_get("pojemnosc"))
            if safe_get("ksztalt"):
                parts.append(safe_get("ksztalt"))
            if safe_get("kierunek_ustawienia"):
                parts.append(safe_get("kierunek_ustawienia"))
            if safe_get("styl"):
                parts.append(f"w stylu {safe_get('styl')}")
            if safe_get("szezlong"):
                parts.append(f"z {safe_get('szezlong')}")
            if safe_get("pikowana"):
                parts.append("pikowana")
                
            # Material
            material_parts = []
            if safe_get("material"):
                material_parts.append(safe_get("material"))
            if safe_get("specyfikacja_materialu"):
                material_parts.append(safe_get("specyfikacja_materialu"))
            if material_parts:
                parts.append(f"wykonana z {' '.join(material_parts)}")
                
            # Size and dimensions
            if safe_get("rozmiar_cm"):
                parts.append(f"o wymiarach {safe_get('rozmiar_cm')}")
            
            # Functions (split from cechy_dodatkowe)
            if safe_get("funkcja_spania"):
                parts.append(f"z funkcją spania {safe_get('funkcja_spania')}")
            if safe_get("schowek"):
                parts.append(f"ze schowkiem {safe_get('schowek')}")
            if safe_get("mechanizm"):
                parts.append(f"z mechanizmem {safe_get('mechanizm')}")
                
            # Color
            if safe_get("kolor"):
                parts.append(f"w kolorze {safe_get('kolor')}")
            if safe_get("podstawa"):
                parts.append(f"na {safe_get('podstawa')}")
            
            # Remaining additional features
            if safe_get("cechy_dodatkowe"):
                parts.append(f"z {safe_get('cechy_dodatkowe')}")
                
            return " ".join(parts)
        
        # Enhanced weighted implementation - ONLY feature values, no filler words
        weighted_features = []
        
        # Build weighted description using only raw feature values
        for feature_key, weight in FEATURE_WEIGHTS.items():
            value = safe_get(feature_key)
            if value:
                repeat_count = int(weight)  # How many times to repeat
                
                # Use only the raw value, no descriptive text
                clean_value = str(value).strip()
                
                # Add the clean value multiple times based on weight
                for _ in range(repeat_count):
                    weighted_features.append(clean_value)
        
        # Join with commas for better token separation
        return ", ".join(weighted_features)
        
    except Exception as e:
        print(f"Error processing features: {e}")
        return "Sofa"

def get_text_embedding(text, dimensions=3072):
    """Get OpenAI text embedding with configurable dimensions"""
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=dimensions  # Support for longer embeddings (up to 3072)
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting text embedding: {e}")
        return None

def l2_normalize(embedding, name="embedding", epsilon=1e-8):
    """
    Normalize embedding to unit L2 norm with verification
    
    Args:
        embedding: numpy array to normalize
        name: name for logging purposes
        epsilon: small value to prevent division by zero
    
    Returns:
        normalized embedding with L2 norm = 1.0
    """
    if embedding is None:
        raise ValueError(f"{name} is None, cannot normalize")
    
    # Convert to numpy if needed
    if torch.is_tensor(embedding):
        embedding = embedding.cpu().numpy()
    
    embedding = np.array(embedding, dtype=np.float32)
    
    # Calculate L2 norm
    norm = np.linalg.norm(embedding)
    
    # Handle zero vector edge case
    if norm < epsilon:
        print(f"Warning: {name} has very small norm ({norm}), using random unit vector")
        embedding = np.random.normal(0, 1, embedding.shape)
        norm = np.linalg.norm(embedding)
    
    # Normalize
    normalized = embedding / (norm + epsilon)
    
    # Verify normalization
    final_norm = np.linalg.norm(normalized)
    
    if abs(final_norm - 1.0) > 1e-6:
        print(f"Warning: {name} normalization failed. Expected norm=1.0, got {final_norm}")
    else:
        print(f"[OK] {name} normalized successfully: norm = {final_norm:.6f}")
    
    return normalized

def combine_embeddings(clip_emb, dinov2_emb, text_emb, weights=None):
    """Combine three embeddings by concatenation with proper L2 normalization"""
    if weights is None:
        weights = EMBEDDING_WEIGHTS
    
    # Ensure all embeddings are properly normalized before combination
    clip_emb = l2_normalize(clip_emb, "CLIP_before_combine")
    dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_before_combine") 
    text_emb = l2_normalize(text_emb, "Text_before_combine")
    
    # Check embedding dimensions
    clip_dim = len(clip_emb)
    dinov2_dim = len(dinov2_emb) 
    text_dim = len(text_emb)
    
    print(f"Embedding dimensions: CLIP={clip_dim}, DINOv2={dinov2_dim}, Text={text_dim}")
    
    # Apply weights by scaling each embedding
    clip_weighted = clip_emb * weights['clip']
    dinov2_weighted = dinov2_emb * weights['dinov2'] 
    text_weighted = text_emb * weights['text']
    
    # Concatenate embeddings (not padding!)
    combined = np.concatenate([clip_weighted, dinov2_weighted, text_weighted])
    
    print(f"Combined embedding dimension: {len(combined)} (should be {clip_dim + dinov2_dim + text_dim})")
    
    # Final normalization
    combined = l2_normalize(combined, "Combined_final")
    
    return combined

def combine_embeddings_weighted_average(clip_emb, dinov2_emb, text_emb, weights=None):
    """OLD METHOD: Combine three embeddings with weighted average after padding"""
    if weights is None:
        weights = EMBEDDING_WEIGHTS
    
    # Ensure all embeddings are properly normalized before combination
    clip_emb = l2_normalize(clip_emb, "CLIP_before_combine")
    dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_before_combine") 
    text_emb = l2_normalize(text_emb, "Text_before_combine")
    
    # Check and align embedding dimensions
    clip_dim = len(clip_emb)
    dinov2_dim = len(dinov2_emb) 
    text_dim = len(text_emb)
    
    # Find maximum dimension for padding
    max_dim = max(clip_dim, dinov2_dim, text_dim)
    
    # Pad smaller embeddings with zeros
    if clip_dim < max_dim:
        clip_emb = np.pad(clip_emb, (0, max_dim - clip_dim), mode='constant')
    
    if dinov2_dim < max_dim:
        dinov2_emb = np.pad(dinov2_emb, (0, max_dim - dinov2_dim), mode='constant')
        
    if text_dim < max_dim:
        text_emb = np.pad(text_emb, (0, max_dim - text_dim), mode='constant')
    
    # Re-normalize after padding
    clip_emb = l2_normalize(clip_emb, "CLIP_after_padding")
    dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_after_padding")
    text_emb = l2_normalize(text_emb, "Text_after_padding")
    
    # Combine with weights
    combined = (weights['clip'] * clip_emb + 
               weights['dinov2'] * dinov2_emb + 
               weights['text'] * text_emb)
    
    # Final normalization with verification
    combined = l2_normalize(combined, "Combined_final")
    
    return combined

def save_faiss_indexes():
    """Save FAISS indexes and metadata to disk"""
    try:
        # Save FAISS indexes
        for index_name, index in faiss_indexes.items():
            if index is not None:
                index_path = os.path.join(FAISS_INDEX_DIR, f"{index_name}.index")
                faiss.write_index(index, index_path)
                print(f"Saved {index_name} index to {index_path}")
        
        # Save metadata
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(product_metadata, f)
        print(f"Saved {len(product_metadata)} products metadata to {METADATA_FILE}")
        
        return True
    except Exception as e:
        print(f"Error saving indexes: {e}")
        return False

def load_faiss_indexes():
    """Load FAISS indexes and metadata from disk"""
    global faiss_indexes, product_metadata
    
    try:
        # Load metadata
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'rb') as f:
                product_metadata = pickle.load(f)
            print(f"Loaded {len(product_metadata)} products metadata")
        
        # Load FAISS indexes
        loaded_count = 0
        for index_name in faiss_indexes.keys():
            index_path = os.path.join(FAISS_INDEX_DIR, f"{index_name}.index")
            if os.path.exists(index_path):
                faiss_indexes[index_name] = faiss.read_index(index_path)
                loaded_count += 1
                print(f"Loaded {index_name} index from {index_path}")
        
        print(f"Loaded {loaded_count} FAISS indexes")
        return True
        
    except Exception as e:
        print(f"Error loading indexes: {e}")
        return False

def find_product_by_id(product_id):
    """Find product index by ID in metadata"""
    for i, metadata in enumerate(product_metadata):
        if metadata.get('id') == product_id:
            return i
    return -1

def remove_product_from_indexes(index_to_remove):
    """Mark product for removal - simplified approach to avoid FAISS reconstruction issues"""
    try:
        # Instead of trying to remove from FAISS indexes (which is complex and error-prone),
        # we'll rely on rebuilding indexes periodically or when needed
        # For now, just log the removal request
        print(f"Product at index {index_to_remove} marked for replacement (FAISS indexes will be rebuilt)")
        return True
    except Exception as e:
        print(f"Error marking for removal: {e}")
        return False

def rebuild_indexes_if_needed():
    """Rebuild FAISS indexes if there are mismatches between metadata and index sizes"""
    try:
        metadata_count = len(product_metadata)
        needs_rebuild = False
        
        for index_name, index in faiss_indexes.items():
            if index is not None:
                if index.ntotal != metadata_count:
                    print(f"Mismatch detected: {index_name} has {index.ntotal} vectors but metadata has {metadata_count} products")
                    needs_rebuild = True
                    break
        
        if needs_rebuild:
            print("Rebuilding FAISS indexes due to size mismatch...")
            # Clear indexes to force rebuild
            for key in faiss_indexes:
                faiss_indexes[key] = None
            return True
        return False
    except Exception as e:
        print(f"Error checking indexes: {e}")
        return False

def process_products_background(task_id, products, index_type):
    """Process products in background thread"""
    with task_lock:
        background_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'total': len(products),
            'message': 'Starting processing...',
            'error': None,
            'result': None
        }
    
    try:
        global product_metadata
        # Clear existing data
        product_metadata = []
        embeddings_data = {'clip': [], 'dinov2': [], 'text': [], 'combined': []}
        
        print(f"[Background] Processing {len(products)} products...")
        
        for i, product in enumerate(products):
            if i % 25 == 0 or i == len(products) - 1:
                progress = (i + 1) / len(products) * 100
                with task_lock:
                    background_tasks[task_id]['progress'] = i + 1
                    background_tasks[task_id]['message'] = f'Processing product {i+1}/{len(products)} ({progress:.1f}%)'
                
                print(f"[Background] Processing product {i+1}/{len(products)} ({progress:.1f}%)")
                
                # Clean GPU memory every 25 products
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            product_id = product.get('id', i)
            
            # Check for existing product with same ID
            existing_index = find_product_by_id(product_id)
            if existing_index >= 0:
                print(f"[Background] Found duplicate product ID {product_id} at index {existing_index}, skipping...")
                continue  # Skip this product instead of trying to remove existing one
            
            # Store metadata with color_group support
            product_metadata.append({
                'id': product_id,
                'features': product.get('features', {}),
                'category': product.get('category', 'sofa'),
                'color_group': product.get('color_group', None)  # New: direct color group
            })
            
            # Get image embedding if provided
            if 'image' in product or 'image_url' in product:
                try:
                    image = get_image_from_data(product)
                except Exception as img_error:
                    print(f"[Background] Skipping product {i+1}: Image error - {str(img_error)}")
                    # Continue to next product without crashing
                    continue
                
                # CLIP embedding with normalization
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_features = clip_model.encode_image(image_input)
                    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                clip_emb = clip_features.cpu().numpy().flatten()
                clip_emb = l2_normalize(clip_emb, f"CLIP_product_{i}")
                embeddings_data['clip'].append(clip_emb)
                
                # DINOv2 embedding with normalization
                dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    dinov2_outputs = dinov2_model(**dinov2_inputs)
                    dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
                dinov2_emb = dinov2_emb.cpu().numpy().flatten()
                dinov2_emb = l2_normalize(dinov2_emb, f"DINOv2_product_{i}")
                embeddings_data['dinov2'].append(dinov2_emb)
                
                # Text embedding with normalization
                if 'text' in product:
                    text = product['text']
                elif 'features' in product:
                    text = features_to_text(product['features'])
                else:
                    text = "Sofa"
                
                text_emb = get_text_embedding(text)
                if text_emb is not None:
                    text_emb = l2_normalize(text_emb, f"Text_product_{i}")
                    embeddings_data['text'].append(text_emb)
                    
                    # Combined embedding with normalization
                    combined_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb)
                    embeddings_data['combined'].append(combined_emb)
        
        # Build FAISS indexes
        with task_lock:
            background_tasks[task_id]['message'] = 'Building FAISS indexes...'
        
        for embed_type, embeddings in embeddings_data.items():
            if embeddings:
                embeddings_array = np.array(embeddings)
                faiss_indexes[embed_type] = create_faiss_index(embeddings_array, index_type)
                print(f"[Background] Built {embed_type} index with {len(embeddings)} vectors")
        
        # Save indexes to disk
        save_success = save_faiss_indexes()
        print(f"[Background] Indexes saved to disk: {save_success}")
        
        # Success
        with task_lock:
            background_tasks[task_id]['status'] = 'completed'
            background_tasks[task_id]['progress'] = len(products)
            background_tasks[task_id]['message'] = 'Processing completed successfully'
            background_tasks[task_id]['result'] = {
                'products_processed': len(products),
                'indexes_built': [k for k, v in faiss_indexes.items() if v is not None]
            }
        
        print(f"[Background] Task {task_id} completed successfully")
        
    except Exception as e:
        print(f"[Background] Task {task_id} failed: {str(e)}")
        with task_lock:
            background_tasks[task_id]['status'] = 'failed'
            background_tasks[task_id]['error'] = str(e)

def create_faiss_index(embeddings, index_type='hnsw'):
    """Create FAISS index from embeddings"""
    dim = embeddings.shape[1]
    
    if index_type == 'hnsw':
        # HNSW index - best for similarity search
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
    elif index_type == 'ivf':
        # IVF index - good for larger datasets
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(100, len(embeddings) // 10))
        index.train(embeddings)
    else:
        # Flat index - exact search
        index = faiss.IndexFlatIP(dim)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    
    return index

def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

def load_image_from_url(image_url, timeout=10):
    """Load image from URL with error handling"""
    try:
        # Validate URL
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme in ['http', 'https']:
            raise ValueError("URL must start with http:// or https://")
        
        # Download image with timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'webp']):
            raise ValueError(f"Invalid content type: {content_type}")
        
        # Load image
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Avoid potential encoding issues in console output
        try:
            print(f"[OK] Loaded image from URL: {image_url[:50]}... Size: {image.size}")
        except UnicodeEncodeError:
            print(f"[OK] Loaded image from URL (encoding issue in display)")
        return image
        
    except requests.exceptions.Timeout:
        raise ValueError(f"Timeout loading image from URL")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error loading image from URL: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing image from URL: {str(e)}")

def remove_background(image, use_cache=True, save_to_disk=True):
    """Remove background from image using Rembg with persistent storage and caching"""
    if not REMBG_AVAILABLE:
        print("Warning: Background removal requested but Rembg not available. Returning original image.")
        return image
    
    try:
        # Generate image hash for consistent caching
        image_hash = get_image_hash(image)
        
        # Check persistent storage first
        if use_cache:
            persistent_image = load_processed_image(image_hash)
            if persistent_image is not None:
                print("Background removal: Persistent storage hit")
                
                # Also cache in memory for faster subsequent access
                background_removal_cache[image_hash] = {
                    'image': persistent_image,
                    'timestamp': time.time()
                }
                return persistent_image
        
        # Check in-memory cache
        if use_cache and image_hash in background_removal_cache:
            cache_entry = background_removal_cache[image_hash]
            # Check TTL
            if time.time() - cache_entry['timestamp'] < BG_CACHE_TTL:
                print("Background removal: Memory cache hit")
                return cache_entry['image']
            else:
                # Remove expired entry
                del background_removal_cache[image_hash]
        
        # Process image - remove background
        print("Background removal: Processing image...")
        start_time = time.time()
        
        # Convert PIL to bytes for rembg
        input_bytes = io.BytesIO()
        image.save(input_bytes, format='PNG')
        input_bytes.seek(0)
        
        # Remove background
        output_bytes = remove(input_bytes.getvalue())
        
        # Convert back to PIL
        output_image = Image.open(io.BytesIO(output_bytes)).convert('RGB')
        
        processing_time = time.time() - start_time
        print(f"Background removal completed in {processing_time:.2f}s")
        
        # Save to persistent storage
        if save_to_disk:
            save_processed_image(image_hash, image, output_image)
        
        # Cache in memory if enabled
        if use_cache:
            # Manage cache size
            if len(background_removal_cache) >= BG_CACHE_MAX_SIZE:
                # Remove oldest entry
                oldest_key = min(background_removal_cache.keys(), 
                               key=lambda k: background_removal_cache[k]['timestamp'])
                del background_removal_cache[oldest_key]
            
            background_removal_cache[image_hash] = {
                'image': output_image,
                'timestamp': time.time()
            }
        
        return output_image
        
    except Exception as e:
        print(f"Background removal failed: {e}. Returning original image.")
        return image

def get_image_from_data(data, remove_bg=False):
    """Get image from either base64 or URL with optional background removal"""
    if 'image_url' in data:
        image = load_image_from_url(data['image_url'])
    elif 'image' in data:
        image = decode_base64_image(data['image'])
    else:
        raise ValueError("Either 'image' (base64) or 'image_url' must be provided")
    
    # Apply background removal if requested
    if remove_bg:
        image = remove_background(image)
    
    return image

def process_image_multiscale(image, scales=[224, 336, 448]):
    """Process image at multiple scales and return averaged embeddings"""
    results = {}
    
    for scale in scales:
        # Resize image to scale while maintaining aspect ratio
        resized_image = image.resize((scale, scale), Image.Resampling.LANCZOS)
        
        # CLIP processing for this scale
        image_input = clip_preprocess(resized_image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(image_input)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_emb = clip_features.cpu().numpy().flatten()
        clip_emb = l2_normalize(clip_emb, f"CLIP_{scale}px")
        
        # DINOv2 processing for this scale
        dinov2_inputs = dinov2_processor(images=resized_image, return_tensors="pt").to(device)
        with torch.no_grad():
            dinov2_outputs = dinov2_model(**dinov2_inputs)
            dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
        dinov2_emb = dinov2_emb.cpu().numpy().flatten()
        dinov2_emb = l2_normalize(dinov2_emb, f"DINOv2_{scale}px")
        
        results[scale] = {
            'clip': clip_emb,
            'dinov2': dinov2_emb
        }
        
        print(f"Processed scale {scale}px: CLIP={len(clip_emb)}, DINOv2={len(dinov2_emb)}")
    
    # Average embeddings across scales
    clip_embeddings = [results[scale]['clip'] for scale in scales]
    dinov2_embeddings = [results[scale]['dinov2'] for scale in scales]
    
    # Ensure all embeddings have same dimension
    max_clip_dim = max(len(emb) for emb in clip_embeddings)
    max_dinov2_dim = max(len(emb) for emb in dinov2_embeddings)
    
    # Pad if necessary
    for i, emb in enumerate(clip_embeddings):
        if len(emb) < max_clip_dim:
            clip_embeddings[i] = np.pad(emb, (0, max_clip_dim - len(emb)), mode='constant')
    
    for i, emb in enumerate(dinov2_embeddings):
        if len(emb) < max_dinov2_dim:
            dinov2_embeddings[i] = np.pad(emb, (0, max_dinov2_dim - len(emb)), mode='constant')
    
    # Average and normalize
    avg_clip = np.mean(clip_embeddings, axis=0)
    avg_dinov2 = np.mean(dinov2_embeddings, axis=0)
    
    avg_clip = l2_normalize(avg_clip, "CLIP_multiscale_avg")
    avg_dinov2 = l2_normalize(avg_dinov2, "DINOv2_multiscale_avg")
    
    print(f"Multi-scale averaging complete: {len(scales)} scales processed")
    
    return avg_clip, avg_dinov2

@app.route('/clip', methods=['POST'])
@require_api_key
def clip_embeddings():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image = get_image_from_data(data)
        
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy and verify normalization
        embeddings_np = image_features.cpu().numpy().flatten()
        embeddings_normalized = l2_normalize(embeddings_np, "CLIP")
        embeddings = embeddings_normalized.tolist()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'embeddings': embeddings,
            'model': 'CLIP ViT-L/14',
            'embedding_dim': len(embeddings)
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/dino', methods=['POST'])
@require_api_key
def dinov2_embeddings():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'image_url' not in data and 'image' not in data:
            return jsonify({'error': 'Either image_url or image data must be provided'}), 400
        
        image = get_image_from_data(data)
        
        inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = dinov2_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Convert to numpy and normalize
        embeddings_np = embeddings.cpu().numpy().flatten()
        embeddings_normalized = l2_normalize(embeddings_np, "DINOv2")
        embeddings = embeddings_normalized.tolist()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'embeddings': embeddings,
            'model': 'DINOv2 ViT-L/14',
            'embedding_dim': len(embeddings)
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/text', methods=['POST'])
@require_api_key
def text_embeddings():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Handle different input formats
        if 'text' in data:
            text = data['text']
        elif 'features' in data:
            # Convert features to text
            text = features_to_text(data['features'])
        else:
            return jsonify({'error': 'No text or features provided'}), 400
        
        # Get text embedding
        embedding = get_text_embedding(text)
        if embedding is None:
            return jsonify({'error': 'Failed to get text embedding'}), 500
        
        # Normalize text embedding
        embedding_normalized = l2_normalize(embedding, "Text")
        
        return jsonify({
            'embeddings': embedding_normalized.tolist(),
            'text': text,
            'model': 'text-embedding-3-large',
            'embedding_dim': len(embedding_normalized)
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/combined', methods=['POST'])
@require_api_key
def combined_embeddings():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Check for required image
        if 'image_url' not in data and 'image' not in data:
            return jsonify({'error': 'Either image_url or image data must be provided'}), 400
        
        # Get custom weights if provided
        weights = data.get('weights', EMBEDDING_WEIGHTS)
        
        # Process image
        image = get_image_from_data(data)
        
        # Get CLIP embedding with normalization
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(image_input)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_emb = clip_features.cpu().numpy().flatten()
        clip_emb = l2_normalize(clip_emb, "CLIP_combined")
        
        # Get DINOv2 embedding with normalization
        dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            dinov2_outputs = dinov2_model(**dinov2_inputs)
            dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
        dinov2_emb = dinov2_emb.cpu().numpy().flatten()
        dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_combined")
        
        # Get text embedding with normalization
        if 'text' in data:
            text = data['text']
        elif 'features' in data:
            text = features_to_text(data['features'])
        else:
            return jsonify({'error': 'No text or features provided for text embedding'}), 400
        
        text_emb = get_text_embedding(text)
        if text_emb is None:
            return jsonify({'error': 'Failed to get text embedding'}), 500
        text_emb = l2_normalize(text_emb, "Text_combined")
        
        # Combine embeddings
        combined_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb, weights)
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'embeddings': combined_emb.tolist(),
            'clip_dim': len(clip_emb),
            'dinov2_dim': len(dinov2_emb),
            'text_dim': len(text_emb),
            'combined_dim': len(combined_emb),
            'weights': weights,
            'text': text,
            'models': 'CLIP ViT-L/14 + DINOv2 ViT-L/14 + text-embedding-3-large'
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/validate-products', methods=['POST'])
@require_api_key
def validate_products():
    """Validate product data before processing"""
    try:
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({'error': 'No products data provided'}), 400
        
        products = data['products']
        validation_report = {
            'total_products': len(products),
            'valid_products': 0,
            'invalid_products': 0,
            'issues': []
        }
        
        for i, product in enumerate(products):
            product_issues = []
            
            # Check image URL/data
            if 'image_url' in product:
                url = product['image_url']
                if not url or not isinstance(url, str):
                    product_issues.append('Empty or invalid image_url')
                elif not url.startswith(('http://', 'https://')):
                    product_issues.append(f'Invalid URL format: {url[:50]}...')
            elif 'image' in product:
                if not product['image']:
                    product_issues.append('Empty image data')
            else:
                product_issues.append('No image_url or image provided')
            
            # Check features
            if 'features' not in product:
                product_issues.append('No features provided')
            
            if product_issues:
                validation_report['invalid_products'] += 1
                validation_report['issues'].append({
                    'product_index': i,
                    'product_id': product.get('id', 'unknown'),
                    'issues': product_issues
                })
            else:
                validation_report['valid_products'] += 1
        
        validation_report['success_rate'] = (validation_report['valid_products'] / len(products)) * 100
        
        return jsonify(validation_report)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/task/<task_id>')
@require_api_key
def get_task_status(task_id):
    """Get status of background task"""
    with task_lock:
        if task_id not in background_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task_info = background_tasks[task_id].copy()
    
    return jsonify({
        'task_id': task_id,
        'status': task_info['status'],
        'progress': task_info['progress'],
        'total': task_info['total'],
        'message': task_info['message'],
        'error': task_info['error'],
        'result': task_info['result']
    })

@app.route('/faiss/save', methods=['POST'])
@require_api_key
def save_indexes():
    """Manually save FAISS indexes to disk"""
    try:
        success = save_faiss_indexes()
        return jsonify({
            'status': 'success' if success else 'error',
            'saved': success,
            'products_count': len(product_metadata),
            'indexes': {k: v is not None for k, v in faiss_indexes.items()}
        })
    except Exception as e:
        return jsonify({'error': f'Save error: {str(e)}'}), 500

@app.route('/debug/check-duplicates', methods=['GET'])
@require_api_key
def check_duplicates():
    """Debug endpoint to check for duplicate product IDs"""
    try:
        product_ids = [meta.get('id') for meta in product_metadata]
        unique_ids = set(product_ids)
        duplicates = []
        
        for product_id in unique_ids:
            count = product_ids.count(product_id)
            if count > 1:
                duplicates.append({'id': product_id, 'count': count})
        
        return jsonify({
            'total_products': len(product_metadata),
            'unique_ids': len(unique_ids),
            'has_duplicates': len(duplicates) > 0,
            'duplicates': duplicates,
            'sample_ids': product_ids[:10]
        })
    except Exception as e:
        return jsonify({'error': f'Check error: {str(e)}'}), 500

@app.route('/faiss/rebuild', methods=['POST'])
@require_api_key
def rebuild_indexes():
    """Rebuild FAISS indexes from existing metadata (emergency recovery)"""
    try:
        if not product_metadata:
            return jsonify({'error': 'No product metadata to rebuild from'}), 400
        
        print(f"Rebuilding indexes from {len(product_metadata)} products...")
        
        # Clear existing indexes
        for key in faiss_indexes:
            faiss_indexes[key] = None
        
        # This will require re-processing all products - which is expensive
        # For now, just load from disk if available
        success = load_faiss_indexes()
        
        return jsonify({
            'status': 'success' if success else 'error',
            'rebuilt': success,
            'products_count': len(product_metadata),
            'message': 'Attempted to reload indexes from disk'
        })
    except Exception as e:
        return jsonify({'error': f'Rebuild error: {str(e)}'}), 500

@app.route('/faiss/rebuild-with-new-models', methods=['POST'])
@require_api_key
def rebuild_indexes_with_new_models():
    """Rebuild FAISS indexes from existing metadata with current (larger) models"""
    try:
        if not product_metadata:
            return jsonify({'error': 'No product metadata to rebuild from'}), 400
        
        print(f"Rebuilding indexes with new models from {len(product_metadata)} products...")
        
        # Clear existing indexes
        global faiss_indexes
        faiss_indexes = {'clip': None, 'dinov2': None, 'text': None, 'combined': None}
        
        # Clear embeddings data
        embeddings_data = {'clip': [], 'dinov2': [], 'text': [], 'combined': []}
        
        # Process products in batches to avoid memory issues
        batch_size = 50
        total_products = len(product_metadata)
        
        for batch_start in range(0, total_products, batch_size):
            batch_end = min(batch_start + batch_size, total_products)
            batch = product_metadata[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_products + batch_size - 1)//batch_size}: products {batch_start+1}-{batch_end}")
            
            for product in batch:
                # Generate embeddings with current (larger) models
                features = product.get('features', {})
                image_url = product.get('image_url', '')
                
                # Generate text embedding
                text_parts = []
                for feature, value in features.items():
                    if value and feature in EMBEDDING_WEIGHTS:
                        text_parts.extend([str(value)] * int(EMBEDDING_WEIGHTS[feature] * 3))
                
                text_input = ', '.join(text_parts) if text_parts else 'sofa'
                text_embedding = get_text_embedding(text_input)
                
                # For image embeddings, we would need to re-download and process images
                # For now, create placeholder embeddings with correct dimensions
                clip_config = MODEL_CONFIG['clip'][CURRENT_MODEL_SIZE]
                dinov2_config = MODEL_CONFIG['dinov2'][CURRENT_MODEL_SIZE]
                
                # Generate placeholder embeddings with correct dimensions
                clip_embedding = np.random.rand(clip_config['dim']).astype(np.float32)
                clip_embedding = clip_embedding / np.linalg.norm(clip_embedding)
                
                dinov2_embedding = np.random.rand(dinov2_config['dim']).astype(np.float32)
                dinov2_embedding = dinov2_embedding / np.linalg.norm(dinov2_embedding)
                
                # Combine embeddings
                combined_embedding = combine_embeddings(clip_embedding, dinov2_embedding, text_embedding)
                
                # Store embeddings
                embeddings_data['clip'].append(clip_embedding)
                embeddings_data['dinov2'].append(dinov2_embedding)
                embeddings_data['text'].append(text_embedding)
                embeddings_data['combined'].append(combined_embedding)
            
            # Clean GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Build FAISS indexes
        print("Building FAISS indexes with new dimensions...")
        
        # CLIP index
        clip_embeddings = np.array(embeddings_data['clip'])
        faiss_indexes['clip'] = create_faiss_index(clip_embeddings, 'hnsw')
        
        # DINOv2 index  
        dinov2_embeddings = np.array(embeddings_data['dinov2'])
        faiss_indexes['dinov2'] = create_faiss_index(dinov2_embeddings, 'hnsw')
        
        # Text index
        text_embeddings = np.array(embeddings_data['text'])
        faiss_indexes['text'] = create_faiss_index(text_embeddings, 'hnsw')
        
        # Combined index
        combined_embeddings = np.array(embeddings_data['combined'])
        faiss_indexes['combined'] = create_faiss_index(combined_embeddings, 'hnsw')
        
        # Save indexes to disk
        save_success = save_faiss_indexes()
        
        return jsonify({
            'status': 'success',
            'products_processed': len(product_metadata),
            'indexes_created': {
                'clip': {'dim': clip_embeddings.shape[1], 'count': clip_embeddings.shape[0]},
                'dinov2': {'dim': dinov2_embeddings.shape[1], 'count': dinov2_embeddings.shape[0]},
                'text': {'dim': text_embeddings.shape[1], 'count': text_embeddings.shape[0]},
                'combined': {'dim': combined_embeddings.shape[1], 'count': combined_embeddings.shape[0]}
            },
            'saved_to_disk': save_success,
            'model_config': {
                'clip': MODEL_CONFIG['clip'][CURRENT_MODEL_SIZE],
                'dinov2': MODEL_CONFIG['dinov2'][CURRENT_MODEL_SIZE]
            },
            'note': 'Image embeddings are placeholders - for full rebuild, use endpoint with actual image processing'
        })
        
    except Exception as e:
        return jsonify({'error': f'Rebuild error: {str(e)}'}), 500

@app.route('/admin/reload-metadata', methods=['POST'])
@require_api_key  
def reload_metadata():
    """Manually reload product metadata from disk"""
    try:
        global product_metadata
        
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'rb') as f:
                product_metadata = pickle.load(f)
            print(f"Manually loaded {len(product_metadata)} products metadata")
            return jsonify({
                'status': 'success',
                'products_loaded': len(product_metadata),
                'message': 'Product metadata reloaded successfully'
            })
        else:
            return jsonify({
                'error': f'Metadata file not found: {METADATA_FILE}'
            }), 404
            
    except Exception as e:
        return jsonify({'error': f'Reload error: {str(e)}'}), 500

@app.route('/stats/processing', methods=['GET'])
@require_api_key
def processing_stats():
    """Get processing statistics and memory usage"""
    try:
        stats = {
            'products_count': len(product_metadata),
            'faiss_indexes': {k: v is not None for k, v in faiss_indexes.items()},
            'active_tasks': len(background_tasks),
            'device': device
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated': f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                'reserved': f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Stats error: {str(e)}'}), 500

@app.route('/colors/groups', methods=['GET'])
@require_api_key
def get_color_groups():
    """Get all available color groups and their colors"""
    try:
        return jsonify({
            'color_groups': COLOR_GROUPS,
            'usage': 'Add "color_filter": "group_name" to search request',
            'example': {
                'embed_type': 'combined',
                'image_url': 'https://...',
                'text': 'sofa',
                'color_filter': 'beżowe',
                'k': 10
            }
        })
    except Exception as e:
        return jsonify({'error': f'Color groups error: {str(e)}'}), 500

@app.route('/colors/analyze', methods=['GET'])
@require_api_key  
def analyze_product_colors():
    """Analyze colors in the current product database"""
    try:
        color_stats = {}
        ungrouped_colors = set()
        
        for metadata in product_metadata:
            color = metadata.get('features', {}).get('kolor', '')
            if color and color != 'null':
                color_group = get_color_group(color)
                if color_group:
                    if color_group not in color_stats:
                        color_stats[color_group] = {'count': 0, 'colors': set()}
                    color_stats[color_group]['count'] += 1
                    color_stats[color_group]['colors'].add(color)
                else:
                    ungrouped_colors.add(color)
        
        # Convert sets to lists for JSON serialization
        for group in color_stats:
            color_stats[group]['colors'] = list(color_stats[group]['colors'])
        
        return jsonify({
            'total_products': len(product_metadata),
            'color_groups_found': color_stats,
            'ungrouped_colors': list(ungrouped_colors),
            'suggestion': 'Consider adding ungrouped colors to COLOR_GROUPS'
        })
    except Exception as e:
        return jsonify({'error': f'Color analysis error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
@public_endpoint
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'clip': clip_model is not None,
            'dinov2': dinov2_model is not None
        },
        'device': device,
        'faiss_indexes': {k: v is not None for k, v in faiss_indexes.items()},
        'products_count': len(product_metadata)
    })

@app.route('/faiss/build', methods=['POST'])
@require_api_key
def build_faiss_index():
    try:
        global faiss_indexes, product_metadata
        
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({'error': 'No products data provided'}), 400
        
        products = data['products']
        index_type = data.get('index_type', 'hnsw')
        
        # Limit batch size to prevent timeouts
        MAX_BATCH_SIZE = 200
        if len(products) > MAX_BATCH_SIZE:
            return jsonify({
                'error': f'Too many products in single batch. Maximum {MAX_BATCH_SIZE}, got {len(products)}. Use /faiss/add for incremental building.',
                'suggested_action': 'Split into smaller batches or use /faiss/add endpoint'
            }), 400
        
        # Clear existing data
        product_metadata = []
        embeddings_data = {'clip': [], 'dinov2': [], 'text': [], 'combined': []}
        
        print(f"Building FAISS index for {len(products)} products (max batch size: {MAX_BATCH_SIZE})...")
        
        for i, product in enumerate(products):
            if i % 50 == 0 or i == len(products) - 1:
                progress = (i + 1) / len(products) * 100
                print(f"Processing product {i+1}/{len(products)} ({progress:.1f}%)")
                
                # Clean GPU memory every 50 products
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Store metadata
            product_metadata.append({
                'id': product.get('id', i),
                'features': product.get('features', {}),
                'category': product.get('category', 'sofa')
            })
            
            # Get image embedding if provided
            if 'image' in product or 'image_url' in product:
                image = get_image_from_data(product)
                
                # CLIP embedding with normalization
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_features = clip_model.encode_image(image_input)
                    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                clip_emb = clip_features.cpu().numpy().flatten()
                clip_emb = l2_normalize(clip_emb, f"CLIP_build_{i}")
                embeddings_data['clip'].append(clip_emb)
                
                # DINOv2 embedding with normalization
                dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    dinov2_outputs = dinov2_model(**dinov2_inputs)
                    dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
                dinov2_emb = dinov2_emb.cpu().numpy().flatten()
                dinov2_emb = l2_normalize(dinov2_emb, f"DINOv2_build_{i}")
                embeddings_data['dinov2'].append(dinov2_emb)
                
                # Text embedding with normalization
                if 'features' in product:
                    text = features_to_text(product['features'])
                    text_emb = get_text_embedding(text)
                    if text_emb is not None:
                        text_emb = l2_normalize(text_emb, f"Text_build_{i}")
                        embeddings_data['text'].append(text_emb)
                        
                        # Combined embedding (already normalized in function)
                        combined_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb)
                        embeddings_data['combined'].append(combined_emb)
                    else:
                        # Create normalized zero vector
                        zero_text = np.zeros(3072)
                        zero_text = l2_normalize(zero_text + 1e-8, f"Text_zero_{i}")  # Add small value to avoid zero norm
                        embeddings_data['text'].append(zero_text)
                        
                        # Combined with zero text embedding
                        combined_emb = combine_embeddings(clip_emb, dinov2_emb, zero_text)
                        embeddings_data['combined'].append(combined_emb)
                
                # GPU cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Build FAISS indexes
        for embed_type, embeddings in embeddings_data.items():
            if embeddings:
                embeddings_array = np.array(embeddings)
                faiss_indexes[embed_type] = create_faiss_index(embeddings_array, index_type)
                print(f"Built {embed_type} index with {len(embeddings)} vectors")
        
        # Save indexes to disk
        save_success = save_faiss_indexes()
        print(f"Indexes saved to disk: {save_success}")
        
        return jsonify({
            'status': 'success',
            'products_indexed': len(products),
            'indexes_built': list(faiss_indexes.keys()),
            'index_type': index_type,
            'dimensions': {k: len(v[0]) if v else 0 for k, v in embeddings_data.items()}
        })
    
    except Exception as e:
        return jsonify({'error': f'Index building error: {str(e)}'}), 500

@app.route('/faiss/build-async', methods=['POST'])
@require_api_key
def build_faiss_index_async():
    """Build FAISS index asynchronously - returns immediately with task_id"""
    try:
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({'error': 'No products data provided'}), 400
        
        products = data['products']
        index_type = data.get('index_type', 'hnsw')
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Start background processing
        thread = threading.Thread(
            target=process_products_background,
            args=(task_id, products, index_type),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'products_count': len(products),
            'message': 'Processing started in background',
            'check_status_url': f'/task/{task_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/faiss/add', methods=['POST'])
@require_api_key
def add_to_faiss_index():
    """Add new products to existing FAISS index"""
    try:
        global faiss_indexes, product_metadata
        
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({'error': 'No products data provided'}), 400
        
        products = data['products']
        append_mode = data.get('append', True)  # Default to append mode
        
        # Limit batch size to prevent timeouts and memory issues
        MAX_BATCH_SIZE = 100  # Reduced from potentially unlimited
        if len(products) > MAX_BATCH_SIZE:
            return jsonify({
                'error': f'Too many products in single batch. Maximum {MAX_BATCH_SIZE}, got {len(products)}. Use /faiss/build-async for larger batches.',
                'suggested_action': 'Split into smaller batches or use /faiss/build-async endpoint'
            }), 400
        
        if not append_mode:
            # Clear existing if not appending
            product_metadata = []
            for key in faiss_indexes:
                faiss_indexes[key] = None
        
        # Store starting index for new products
        start_index = len(product_metadata)
        embeddings_data = {'clip': [], 'dinov2': [], 'text': [], 'combined': []}
        
        print(f"Adding {len(products)} products to FAISS index (append={append_mode})...")
        
        for i, product in enumerate(products):
            if i % 25 == 0:  # More frequent logging
                print(f"Processing product {i+1}/{len(products)}")
                # Clean GPU memory more frequently
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            product_id = product.get('id', start_index + i)
            
            # Check for existing product with same ID
            existing_index = find_product_by_id(product_id)
            if existing_index >= 0:
                print(f"Found duplicate product ID {product_id} at index {existing_index}, skipping...")
                continue  # Skip this product instead of trying to remove existing one
            
            # Store metadata with color_group support
            product_metadata.append({
                'id': product_id,
                'features': product.get('features', {}),
                'category': product.get('category', 'sofa'),
                'color_group': product.get('color_group', None)  # New: direct color group
            })
            
            # Get image embedding if provided
            if 'image' in product or 'image_url' in product:
                image = get_image_from_data(product)
                
                # CLIP embedding with normalization
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_features = clip_model.encode_image(image_input)
                    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                clip_emb = clip_features.cpu().numpy().flatten()
                clip_emb = l2_normalize(clip_emb, f"CLIP_add_{start_index + i}")
                embeddings_data['clip'].append(clip_emb)
                
                # DINOv2 embedding with normalization
                dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    dinov2_outputs = dinov2_model(**dinov2_inputs)
                    dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
                dinov2_emb = dinov2_emb.cpu().numpy().flatten()
                dinov2_emb = l2_normalize(dinov2_emb, f"DINOv2_add_{start_index + i}")
                embeddings_data['dinov2'].append(dinov2_emb)
                
                # Text embedding with normalization
                if 'features' in product:
                    text = features_to_text(product['features'])
                    text_emb = get_text_embedding(text)
                    if text_emb is not None:
                        text_emb = l2_normalize(text_emb, f"Text_add_{start_index + i}")
                        embeddings_data['text'].append(text_emb)
                        
                        # Combined embedding (already normalized in function)
                        combined_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb)
                        embeddings_data['combined'].append(combined_emb)
                    else:
                        # Create normalized zero vector
                        zero_text = np.zeros(3072)
                        zero_text = l2_normalize(zero_text + 1e-8, f"Text_zero_add_{start_index + i}")
                        embeddings_data['text'].append(zero_text)
                        
                        # Combined with zero text embedding
                        combined_emb = combine_embeddings(clip_emb, dinov2_emb, zero_text)
                        embeddings_data['combined'].append(combined_emb)
                
                # GPU cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Add to existing indexes or create new ones
        for embed_type, new_embeddings in embeddings_data.items():
            if new_embeddings:
                new_embeddings_array = np.array(new_embeddings).astype('float32')
                
                if faiss_indexes[embed_type] is None or not append_mode:
                    # Create new index
                    faiss_indexes[embed_type] = create_faiss_index(new_embeddings_array, data.get('index_type', 'hnsw'))
                    print(f"Created new {embed_type} index with {len(new_embeddings)} vectors")
                else:
                    # Add to existing index
                    faiss_indexes[embed_type].add(new_embeddings_array)
                    print(f"Added {len(new_embeddings)} vectors to existing {embed_type} index")
        
        # Save indexes to disk
        save_success = save_faiss_indexes()
        print(f"Indexes saved to disk: {save_success}")
        
        return jsonify({
            'status': 'success',
            'products_added': len(products),
            'total_products': len(product_metadata),
            'append_mode': append_mode,
            'indexes_updated': list(embeddings_data.keys()),
            'dimensions': {k: len(v[0]) if v else 0 for k, v in embeddings_data.items()}
        })
    
    except Exception as e:
        return jsonify({'error': f'Index adding error: {str(e)}'}), 500

def process_add_products_background(task_id, products, append_mode):
    """Process products in background thread for add-async endpoint"""
    with task_lock:
        background_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'total': len(products),
            'message': 'Starting processing...',
            'error': None,
            'result': None
        }
    
    try:
        global faiss_indexes, product_metadata
        
        print(f"[Background Add] Starting with {len(product_metadata)} existing products, append_mode={append_mode}")
        
        if not append_mode:
            # Clear existing if not appending
            print(f"[Background Add] CLEARING existing {len(product_metadata)} products (append_mode=False)")
            product_metadata = []
            for key in faiss_indexes:
                faiss_indexes[key] = None
        else:
            print(f"[Background Add] KEEPING existing {len(product_metadata)} products (append_mode=True)")
        
        # Store starting index for new products
        start_index = len(product_metadata)
        embeddings_data = {'clip': [], 'dinov2': [], 'text': [], 'combined': []}
        
        print(f"[Background Add] Processing {len(products)} products (append={append_mode})...")
        
        for i, product in enumerate(products):
            if i % 25 == 0 or i == len(products) - 1:
                progress = (i + 1) / len(products) * 100
                with task_lock:
                    background_tasks[task_id]['progress'] = i + 1
                    background_tasks[task_id]['message'] = f'Processing product {i+1}/{len(products)} ({progress:.1f}%)'
                
                print(f"[Background Add] Processing product {i+1}/{len(products)} ({progress:.1f}%)")
                
                # Clean GPU memory every 25 products
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            product_id = product.get('id', start_index + i)
            
            # Check for existing product with same ID
            existing_index = find_product_by_id(product_id)
            if existing_index >= 0:
                print(f"[Background Add] Found duplicate product ID {product_id} at index {existing_index}, skipping...")
                continue  # Skip this product instead of trying to remove existing one
            
            # Store metadata with color_group support
            product_metadata.append({
                'id': product_id,
                'features': product.get('features', {}),
                'category': product.get('category', 'sofa'),
                'color_group': product.get('color_group', None)  # New: direct color group
            })
            
            # Get image embedding if provided
            if 'image' in product or 'image_url' in product:
                try:
                    image = get_image_from_data(product)
                    
                    # CLIP embedding with normalization
                    image_input = clip_preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        clip_features = clip_model.encode_image(image_input)
                        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                    clip_emb = clip_features.cpu().numpy().flatten()
                    clip_emb = l2_normalize(clip_emb, f"CLIP_add_bg_{start_index + i}")
                    embeddings_data['clip'].append(clip_emb)
                    
                    # DINOv2 embedding with normalization
                    dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        dinov2_outputs = dinov2_model(**dinov2_inputs)
                        dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
                    dinov2_emb = dinov2_emb.cpu().numpy().flatten()
                    dinov2_emb = l2_normalize(dinov2_emb, f"DINOv2_add_bg_{start_index + i}")
                    embeddings_data['dinov2'].append(dinov2_emb)
                    
                    # Text embedding with normalization
                    if 'features' in product:
                        text = features_to_text(product['features'])
                        text_emb = get_text_embedding(text)
                        if text_emb is not None:
                            text_emb = l2_normalize(text_emb, f"Text_add_bg_{start_index + i}")
                            embeddings_data['text'].append(text_emb)
                            
                            # Combined embedding (already normalized in function)
                            combined_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb)
                            embeddings_data['combined'].append(combined_emb)
                        else:
                            # Create normalized zero vector
                            zero_text = np.zeros(3072)
                            zero_text = l2_normalize(zero_text + 1e-8, f"Text_zero_add_bg_{start_index + i}")
                            embeddings_data['text'].append(zero_text)
                            
                            # Combined with zero text embedding
                            combined_emb = combine_embeddings(clip_emb, dinov2_emb, zero_text)
                            embeddings_data['combined'].append(combined_emb)
                    
                    # GPU cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"[Background Add] Error processing product {i}: {e}")
                    continue
        
        # Add to existing indexes or create new ones
        with task_lock:
            background_tasks[task_id]['message'] = 'Building FAISS indexes...'
        
        for embed_type, new_embeddings in embeddings_data.items():
            if new_embeddings:
                new_embeddings_array = np.array(new_embeddings).astype('float32')
                
                if faiss_indexes[embed_type] is None or not append_mode:
                    # Create new index
                    faiss_indexes[embed_type] = create_faiss_index(new_embeddings_array, 'hnsw')
                    print(f"[Background Add] Created new {embed_type} index with {len(new_embeddings)} vectors")
                else:
                    # Add to existing index
                    faiss_indexes[embed_type].add(new_embeddings_array)
                    print(f"[Background Add] Added {len(new_embeddings)} vectors to existing {embed_type} index")
        
        # Save indexes to disk
        save_success = save_faiss_indexes()
        print(f"[Background Add] Indexes saved to disk: {save_success}")
        
        # Success
        with task_lock:
            background_tasks[task_id]['status'] = 'completed'
            background_tasks[task_id]['progress'] = len(products)
            background_tasks[task_id]['message'] = 'Processing completed successfully'
            background_tasks[task_id]['result'] = {
                'products_count': len(products),
                'total_products': len(product_metadata),
                'indexes_updated': [k for k, v in faiss_indexes.items() if v is not None],
                'append_mode': append_mode
            }
        
    except Exception as e:
        print(f"[Background Add] Error: {e}")
        with task_lock:
            background_tasks[task_id]['status'] = 'error'
            background_tasks[task_id]['error'] = str(e)
            background_tasks[task_id]['message'] = f'Error: {str(e)}'

@app.route('/faiss/add-async', methods=['POST'])
@require_api_key
def add_to_faiss_index_async():
    """Add new products to existing FAISS index asynchronously"""
    try:
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({'error': 'No products data provided'}), 400
        
        products = data['products']
        # FORCE append mode for add-async endpoint - it should NEVER reset
        append_mode = True  # Always append for add-async
        
        # Debug logging - show ALL possible append parameters
        print(f"[add-async] Received request: products={len(products)}, FORCED append_mode={append_mode}")
        print(f"[add-async] Current products before processing: {len(product_metadata)}")
        print(f"[add-async] Raw 'append' from request: {data.get('append', 'NOT_PROVIDED')}")
        print(f"[add-async] Raw 'append_mode' from request: {data.get('append_mode', 'NOT_PROVIDED')}")
        print(f"[add-async] ALL request keys: {list(data.keys())}")
        print(f"[add-async] FORCING append_mode=True regardless of input")
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Start background processing
        thread = threading.Thread(
            target=process_add_products_background,
            args=(task_id, products, append_mode),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'products_count': len(products),
            'append_mode': append_mode,
            'message': 'Adding products started in background',
            'check_status_url': f'/task/{task_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/faiss/search', methods=['POST'])
@require_api_key
def search_faiss():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        embed_type = data.get('embed_type', 'combined')
        k = data.get('k', 10)
        color_filter = data.get('color_filter', None)  # New: color group filter
        
        if faiss_indexes[embed_type] is None:
            return jsonify({'error': f'No {embed_type} index available'}), 400
        
        # Get query embedding
        if embed_type == 'combined':
            if 'image_url' not in data and 'image' not in data:
                return jsonify({'error': 'Either image_url or image data required for combined search'}), 400
            
            # Get all embeddings for combination
            image = get_image_from_data(data)
            
            # CLIP with normalization
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_features = clip_model.encode_image(image_input)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_emb = clip_features.cpu().numpy().flatten()
            clip_emb = l2_normalize(clip_emb, "CLIP_query")
            
            # DINOv2 with normalization
            dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                dinov2_outputs = dinov2_model(**dinov2_inputs)
                dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
            dinov2_emb = dinov2_emb.cpu().numpy().flatten()
            dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_query")
            
            # Text with normalization and optional feature weighting
            use_weighted_features = data.get('use_weighted_features', False)
            use_enhanced_weights = data.get('use_enhanced_weights', False)
            
            if 'text' in data:
                text = data['text']
            elif 'features' in data:
                text = features_to_text(data['features'], use_weighted=use_weighted_features)
            else:
                return jsonify({'error': 'Text or features required for combined search'}), 400
            
            text_emb = get_text_embedding(text)
            if text_emb is None:
                return jsonify({'error': 'Failed to get text embedding'}), 500
            text_emb = l2_normalize(text_emb, "Text_query")
            
            # Combine with proper normalization and enhanced weights if requested
            if use_enhanced_weights:
                weights = EMBEDDING_WEIGHTS_ENHANCED
            else:
                weights = data.get('weights', EMBEDDING_WEIGHTS)
            query_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb, weights)
            
        elif embed_type == 'clip' or embed_type == 'dinov2':
            if 'image_url' not in data and 'image' not in data:
                return jsonify({'error': 'Either image_url or image data required for visual search'}), 400
            
            image = get_image_from_data(data)
            
            if embed_type == 'clip':
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_features = clip_model.encode_image(image_input)
                    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                query_emb = clip_features.cpu().numpy().flatten()
                query_emb = l2_normalize(query_emb, "CLIP_search_query")
            else:  # dinov2
                dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    dinov2_outputs = dinov2_model(**dinov2_inputs)
                    query_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
                query_emb = query_emb.cpu().numpy().flatten()
                query_emb = l2_normalize(query_emb, "DINOv2_search_query")
                
        elif embed_type == 'text':
            if 'text' in data:
                text = data['text']
            elif 'features' in data:
                text = features_to_text(data['features'])
            else:
                return jsonify({'error': 'Text or features required for text search'}), 400
            
            query_emb = get_text_embedding(text)
            if query_emb is None:
                return jsonify({'error': 'Failed to get text embedding'}), 500
            query_emb = l2_normalize(query_emb, "Text_search_query")
        
        # Search with color filtering
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        # If color filter is specified, search more results to filter
        search_k = k * 5 if color_filter else k  # Get 5x more results to filter
        distances, indices = faiss_indexes[embed_type].search(query_emb, search_k)
        
        # Prepare results with optional color filtering
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(product_metadata):
                # Apply color filter if specified
                if color_filter:
                    # Use direct color_group if available, otherwise derive from color
                    product_color_group = product_metadata[idx].get('color_group')
                    if not product_color_group:
                        product_color = product_metadata[idx]['features'].get('kolor', '')
                        product_color_group = get_color_group(product_color)
                    
                    if product_color_group != color_filter:
                        continue  # Skip products that don't match color group
                
                # Convert distance to similarity: lower distance = higher similarity
                similarity = 1.0 - float(distance)
                result = {
                    'rank': len(results) + 1,  # Re-rank after filtering
                    'similarity': similarity,
                    'product_id': product_metadata[idx]['id'],
                    'features': product_metadata[idx]['features'],
                    'category': product_metadata[idx]['category'],
                    'color_group': product_metadata[idx].get('color_group')  # Include color_group
                }
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'results': results,
            'embed_type': embed_type,
            'k': k,
            'total_indexed': len(product_metadata)
        })
    
    except Exception as e:
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/faiss/clear', methods=['POST'])
@require_api_key
def clear_faiss_indexes():
    """Clear all FAISS indexes and metadata"""
    try:
        global faiss_indexes, product_metadata
        
        # Clear all indexes
        for key in faiss_indexes:
            faiss_indexes[key] = None
        
        # Clear metadata
        product_metadata = []
        
        # Remove index files from disk
        import os
        for filename in os.listdir(FAISS_INDEX_DIR):
            if filename.endswith('.index'):
                file_path = os.path.join(FAISS_INDEX_DIR, filename)
                os.remove(file_path)
                print(f"Removed {file_path}")
        
        # Remove metadata file
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
            print(f"Removed {METADATA_FILE}")
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'status': 'success',
            'message': 'All FAISS indexes and metadata cleared',
            'indexes_cleared': list(faiss_indexes.keys())
        })
        
    except Exception as e:
        return jsonify({'error': f'Clear error: {str(e)}'}), 500

@app.route('/faiss/stats', methods=['GET'])
@public_endpoint
def faiss_stats():
    try:
        stats = {}
        for embed_type, index in faiss_indexes.items():
            if index is not None:
                stats[embed_type] = {
                    'ntotal': index.ntotal,
                    'dimension': index.d,
                    'is_trained': index.is_trained
                }
            else:
                stats[embed_type] = None
        
        return jsonify({
            'indexes': stats,
            'products_count': len(product_metadata),
            'embedding_weights': EMBEDDING_WEIGHTS
        })
    
    except Exception as e:
        return jsonify({'error': f'Stats error: {str(e)}'}), 500

@app.route('/test/rnn', methods=['POST'])
@require_api_key
def test_reciprocal_nearest_neighbors():
    """Test Reciprocal Nearest Neighbors - check if A->B and B->A are mutual"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        embed_type = data.get('embed_type', 'combined')
        k = data.get('k', 10)
        test_samples = data.get('test_samples', 50)  # Number of random samples to test
        
        if faiss_indexes[embed_type] is None:
            return jsonify({'error': f'No {embed_type} index available'}), 400
        
        if len(product_metadata) < test_samples:
            test_samples = len(product_metadata)
        
        # Random sample of products for testing
        import random
        test_indices = random.sample(range(len(product_metadata)), test_samples)
        
        rnn_results = []
        total_pairs = 0
        reciprocal_pairs = 0
        
        print(f"Testing RNN with {test_samples} samples...")
        
        for i, test_idx in enumerate(test_indices):
            if i % 10 == 0:
                print(f"Testing sample {i+1}/{test_samples}")
            
            # Get embeddings for test product
            if embed_type == 'combined':
                # Use stored combined embedding (approximation)
                test_emb = np.array(faiss_indexes[embed_type].reconstruct(test_idx)).reshape(1, -1)
            else:
                # Use stored embedding
                test_emb = np.array(faiss_indexes[embed_type].reconstruct(test_idx)).reshape(1, -1)
            
            # Find k nearest neighbors
            distances, indices = faiss_indexes[embed_type].search(test_emb.astype('float32'), k + 1)
            
            # Skip self (first result)
            neighbors = indices[0][1:k+1]
            
            # Test reciprocal relationship
            for neighbor_idx in neighbors:
                total_pairs += 1
                
                # Get neighbor's embedding
                neighbor_emb = np.array(faiss_indexes[embed_type].reconstruct(neighbor_idx)).reshape(1, -1)
                
                # Find neighbor's nearest neighbors
                neighbor_distances, neighbor_indices = faiss_indexes[embed_type].search(
                    neighbor_emb.astype('float32'), k + 1
                )
                
                # Check if original test_idx is in neighbor's k nearest neighbors
                neighbor_neighbors = neighbor_indices[0][1:k+1]  # Skip self
                
                if test_idx in neighbor_neighbors:
                    reciprocal_pairs += 1
                    rnn_results.append({
                        'query_idx': int(test_idx),
                        'neighbor_idx': int(neighbor_idx),
                        'reciprocal': True,
                        'query_product': product_metadata[test_idx]['id'],
                        'neighbor_product': product_metadata[neighbor_idx]['id']
                    })
        
        # Calculate metrics
        precision = reciprocal_pairs / total_pairs if total_pairs > 0 else 0
        recall = precision  # In this context, precision == recall for RNN
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return jsonify({
            'test_results': {
                'total_pairs_tested': total_pairs,
                'reciprocal_pairs': reciprocal_pairs,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'reciprocal_rate': precision
            },
            'embed_type': embed_type,
            'k': k,
            'test_samples': test_samples,
            'reciprocal_examples': rnn_results[:10]  # Show first 10 examples
        })
    
    except Exception as e:
        return jsonify({'error': f'RNN test error: {str(e)}'}), 500

@app.route('/test/rnn-enhanced', methods=['POST'])
@require_api_key
def test_rnn_enhanced():
    """Enhanced RNN test with exact NN option and configurable thresholds"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        embed_type = data.get('embed_type', 'combined')
        k = data.get('k', 10)
        test_samples = data.get('test_samples', 50)
        use_exact_nn = data.get('use_exact_nn', False)  # Use exact NN instead of FAISS
        similarity_threshold = data.get('similarity_threshold', 0.7)  # Threshold for normalized space
        quality_mode = data.get('quality_mode', False)  # High quality mode with more tests
        
        if faiss_indexes[embed_type] is None:
            return jsonify({'error': f'No {embed_type} index available'}), 400
        
        if len(product_metadata) < test_samples:
            test_samples = len(product_metadata)
        
        # Quality mode: test more samples but fewer neighbors
        if quality_mode:
            test_samples = min(test_samples * 2, len(product_metadata))
            k = min(k, 5)  # Fewer neighbors for quality assessment
        
        # Random sample of products for testing
        import random
        test_indices = random.sample(range(len(product_metadata)), test_samples)
        
        rnn_results = []
        total_pairs = 0
        reciprocal_pairs = 0
        above_threshold_pairs = 0
        similarity_scores = []
        
        print(f"Enhanced RNN test: {test_samples} samples, k={k}, exact_nn={use_exact_nn}, threshold={similarity_threshold}")
        
        if use_exact_nn:
            # Get all embeddings for exact NN
            all_embeddings = []
            for i in range(faiss_indexes[embed_type].ntotal):
                emb = faiss_indexes[embed_type].reconstruct(i)
                all_embeddings.append(emb)
            all_embeddings = np.array(all_embeddings)
            print(f"Loaded {len(all_embeddings)} embeddings for exact NN")
        
        for i, test_idx in enumerate(test_indices):
            if i % 10 == 0:
                print(f"Testing sample {i+1}/{test_samples}")
            
            # Get embeddings for test product
            test_emb = np.array(faiss_indexes[embed_type].reconstruct(test_idx)).reshape(1, -1)
            
            if use_exact_nn:
                # Exact NN: compute similarities to all products
                similarities = cosine_similarity(test_emb, all_embeddings)[0]
                # Get top k+1 (excluding self)
                top_indices = np.argsort(similarities)[::-1][:k+1]
                # Remove self if present
                neighbors = [idx for idx in top_indices if idx != test_idx][:k]
                neighbor_similarities = [similarities[idx] for idx in neighbors]
            else:
                # FAISS approximate NN
                distances, indices = faiss_indexes[embed_type].search(test_emb.astype('float32'), k + 1)
                neighbors = indices[0][1:k+1]  # Skip self
                # Convert FAISS distances to similarities (for cosine: sim = 1 - dist/2)
                neighbor_similarities = [max(0, 1 - dist/2) for dist in distances[0][1:k+1]]
            
            # Test reciprocal relationship
            for j, neighbor_idx in enumerate(neighbors):
                total_pairs += 1
                neighbor_similarity = neighbor_similarities[j]
                similarity_scores.append(neighbor_similarity)
                
                # Check similarity threshold
                if neighbor_similarity >= similarity_threshold:
                    above_threshold_pairs += 1
                
                # Get neighbor's embedding
                neighbor_emb = np.array(faiss_indexes[embed_type].reconstruct(neighbor_idx)).reshape(1, -1)
                
                if use_exact_nn:
                    # Exact NN for neighbor
                    neighbor_similarities_all = cosine_similarity(neighbor_emb, all_embeddings)[0]
                    neighbor_top_indices = np.argsort(neighbor_similarities_all)[::-1][:k+1]
                    neighbor_neighbors = [idx for idx in neighbor_top_indices if idx != neighbor_idx][:k]
                else:
                    # FAISS NN for neighbor
                    neighbor_distances, neighbor_indices = faiss_indexes[embed_type].search(
                        neighbor_emb.astype('float32'), k + 1
                    )
                    neighbor_neighbors = neighbor_indices[0][1:k+1]  # Skip self
                
                # Check reciprocal relationship
                is_reciprocal = test_idx in neighbor_neighbors
                if is_reciprocal:
                    reciprocal_pairs += 1
                
                # Store detailed result for quality mode
                if quality_mode or is_reciprocal:
                    rnn_results.append({
                        'query_idx': int(test_idx),
                        'neighbor_idx': int(neighbor_idx),
                        'similarity': float(neighbor_similarity),
                        'above_threshold': neighbor_similarity >= similarity_threshold,
                        'reciprocal': is_reciprocal,
                        'query_product': product_metadata[test_idx]['id'],
                        'neighbor_product': product_metadata[neighbor_idx]['id']
                    })
        
        # Calculate metrics
        precision = reciprocal_pairs / total_pairs if total_pairs > 0 else 0
        recall = precision  # In RNN context
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        threshold_rate = above_threshold_pairs / total_pairs if total_pairs > 0 else 0
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        return jsonify({
            'test_results': {
                'total_pairs_tested': total_pairs,
                'reciprocal_pairs': reciprocal_pairs,
                'above_threshold_pairs': above_threshold_pairs,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'reciprocal_rate': precision,
                'threshold_rate': threshold_rate,
                'avg_similarity': float(avg_similarity),
                'similarity_stats': {
                    'min': float(np.min(similarity_scores)) if similarity_scores else 0,
                    'max': float(np.max(similarity_scores)) if similarity_scores else 0,
                    'std': float(np.std(similarity_scores)) if similarity_scores else 0
                }
            },
            'configuration': {
                'embed_type': embed_type,
                'k': k,
                'test_samples': test_samples,
                'use_exact_nn': use_exact_nn,
                'similarity_threshold': similarity_threshold,
                'quality_mode': quality_mode
            },
            'reciprocal_examples': rnn_results[:20] if quality_mode else rnn_results[:10]
        })
    
    except Exception as e:
        return jsonify({'error': f'Enhanced RNN test error: {str(e)}'}), 500

@app.route('/batch', methods=['POST'])
@require_api_key
def batch_process():
    """Process multiple products in batch"""
    try:
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({'error': 'No products data provided'}), 400
        
        products = data['products']
        embed_types = data.get('embed_types', ['combined'])
        
        results = []
        
        print(f"Batch processing {len(products)} products...")
        
        for i, product in enumerate(products):
            if i % 10 == 0:
                print(f"Processing product {i+1}/{len(products)}")
            
            product_result = {
                'id': product.get('id', i),
                'embeddings': {}
            }
            
            if 'image' in product or 'image_url' in product:
                image = get_image_from_data(product)
                
                # Get embeddings based on requested types
                if 'clip' in embed_types:
                    image_input = clip_preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        clip_features = clip_model.encode_image(image_input)
                        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                    product_result['embeddings']['clip'] = clip_features.cpu().numpy().flatten().tolist()
                
                if 'dinov2' in embed_types:
                    dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        dinov2_outputs = dinov2_model(**dinov2_inputs)
                        dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
                    product_result['embeddings']['dinov2'] = dinov2_emb.cpu().numpy().flatten().tolist()
                
                if 'text' in embed_types and 'features' in product:
                    text = features_to_text(product['features'])
                    text_emb = get_text_embedding(text)
                    if text_emb is not None:
                        product_result['embeddings']['text'] = text_emb.tolist()
                    product_result['text'] = text
                
                if 'combined' in embed_types and all(k in product_result['embeddings'] for k in ['clip', 'dinov2', 'text']):
                    clip_emb = np.array(product_result['embeddings']['clip'])
                    dinov2_emb = np.array(product_result['embeddings']['dinov2'])
                    text_emb = np.array(product_result['embeddings']['text'])
                    
                    combined_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb)
                    product_result['embeddings']['combined'] = combined_emb.tolist()
                
                # GPU cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            results.append(product_result)
        
        return jsonify({
            'results': results,
            'processed_count': len(products),
            'embed_types': embed_types
        })
    
    except Exception as e:
        return jsonify({'error': f'Batch processing error: {str(e)}'}), 500

@app.route('/debug/verify-norm', methods=['POST'])
@require_api_key
def verify_norm():
    """Debug endpoint to verify L2 normalization of embeddings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'image_url' not in data and 'image' not in data:
            return jsonify({'error': 'Either image_url or image data required for verification'}), 400
        
        results = {}
        image = get_image_from_data(data)
        
        # Test CLIP normalization
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(image_input)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_emb = clip_features.cpu().numpy().flatten()
        
        clip_norm_before = np.linalg.norm(clip_emb)
        clip_emb_normalized = l2_normalize(clip_emb, "CLIP_debug")
        clip_norm_after = np.linalg.norm(clip_emb_normalized)
        
        results['clip'] = {
            'norm_before': float(clip_norm_before),
            'norm_after': float(clip_norm_after),
            'is_normalized': abs(clip_norm_after - 1.0) < 1e-6
        }
        
        # Test DINOv2 normalization
        dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            dinov2_outputs = dinov2_model(**dinov2_inputs)
            dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
        dinov2_emb = dinov2_emb.cpu().numpy().flatten()
        
        dinov2_norm_before = np.linalg.norm(dinov2_emb)
        dinov2_emb_normalized = l2_normalize(dinov2_emb, "DINOv2_debug")
        dinov2_norm_after = np.linalg.norm(dinov2_emb_normalized)
        
        results['dinov2'] = {
            'norm_before': float(dinov2_norm_before),
            'norm_after': float(dinov2_norm_after),
            'is_normalized': abs(dinov2_norm_after - 1.0) < 1e-6
        }
        
        # Test Text normalization
        if 'features' in data:
            text = features_to_text(data['features'])
            text_emb = get_text_embedding(text)
            if text_emb is not None:
                text_norm_before = np.linalg.norm(text_emb)
                text_emb_normalized = l2_normalize(text_emb, "Text_debug")
                text_norm_after = np.linalg.norm(text_emb_normalized)
                
                results['text'] = {
                    'norm_before': float(text_norm_before),
                    'norm_after': float(text_norm_after),
                    'is_normalized': abs(text_norm_after - 1.0) < 1e-6,
                    'text_used': text
                }
                
                # Test Combined normalization
                combined_emb = combine_embeddings(clip_emb_normalized, dinov2_emb_normalized, text_emb_normalized)
                combined_norm = np.linalg.norm(combined_emb)
                
                results['combined'] = {
                    'norm': float(combined_norm),
                    'is_normalized': abs(combined_norm - 1.0) < 1e-6,
                    'weights_used': EMBEDDING_WEIGHTS
                }
        
        # Overall verification
        all_normalized = all(
            result.get('is_normalized', False) 
            for result in results.values()
        )
        
        results['summary'] = {
            'all_embeddings_normalized': all_normalized,
            'total_checked': len(results) - 1,  # Exclude summary itself
            'epsilon_threshold': 1e-6
        }
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Norm verification error: {str(e)}'}), 500

@app.route('/faiss/search/exact', methods=['POST'])
@require_api_key
def faiss_search_exact():
    """Exact nearest neighbor search (brute force) for comparison with approximate search"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        embed_type = data.get('embed_type', 'combined')
        k = data.get('k', 10)
        color_filter = data.get('color_filter')  # Optional color group filter
        use_enhanced_weights = data.get('use_enhanced_weights', False)
        use_weighted_features = data.get('use_weighted_features', True)
        
        if faiss_indexes[embed_type] is None:
            return jsonify({'error': f'No {embed_type} index available'}), 400
        
        # Get query embedding (same logic as regular search)
        if embed_type == 'combined':
            if 'image_url' not in data and 'image' not in data:
                return jsonify({'error': 'Either image_url or image data required for combined search'}), 400
            
            image = get_image_from_data(data)
            
            # CLIP with normalization
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_features = clip_model.encode_image(image_input)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_emb = clip_features.cpu().numpy().flatten()
            clip_emb = l2_normalize(clip_emb, "CLIP_query")
            
            # DINOv2 with normalization
            dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                dinov2_outputs = dinov2_model(**dinov2_inputs)
                dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
            dinov2_emb = dinov2_emb.cpu().numpy().flatten()
            dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_query")
            
            # Text with normalization and weighting
            if 'text' in data:
                text = data['text']
            elif 'features' in data:
                text = features_to_text(data['features'], use_weighted=use_weighted_features)
            else:
                return jsonify({'error': 'Text or features required for combined search'}), 400
            
            text_emb = get_text_embedding(text)
            if text_emb is None:
                return jsonify({'error': 'Failed to get text embedding'}), 500
            text_emb = l2_normalize(text_emb, "Text_query")
            
            # Combine with proper weights
            weights = EMBEDDING_WEIGHTS_ENHANCED if use_enhanced_weights else data.get('weights', EMBEDDING_WEIGHTS)
            query_emb = combine_embeddings(clip_emb, dinov2_emb, text_emb, weights)
            
        else:
            return jsonify({'error': 'Exact search currently only supports combined embeddings'}), 400
        
        # EXACT SEARCH: Compute distances to ALL products manually
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        # Get all embeddings from index
        all_embeddings = []
        for i in range(faiss_indexes[embed_type].ntotal):
            emb = faiss_indexes[embed_type].reconstruct(i)
            all_embeddings.append(emb)
        
        all_embeddings = np.array(all_embeddings)
        
        # Compute exact L2 distances
        from scipy.spatial.distance import cdist
        distances = cdist(query_emb, all_embeddings, metric='euclidean')[0]
        
        # Get sorted indices
        sorted_indices = np.argsort(distances)
        
        # Prepare results with optional color filtering
        results = []
        search_k = k * 5 if color_filter else k
        
        for idx in sorted_indices[:search_k]:
            if idx < len(product_metadata):
                # Apply color filter if specified
                if color_filter:
                    product_color = product_metadata[idx]['features'].get('kolor', '')
                    product_color_group = get_color_group(product_color)
                    if product_color_group != color_filter:
                        continue
                
                similarity = 1.0 - float(distances[idx])
                result = {
                    'rank': len(results) + 1,
                    'similarity': similarity,
                    'exact_distance': float(distances[idx]),
                    'product_id': product_metadata[idx]['id'],
                    'features': product_metadata[idx]['features'],
                    'category': product_metadata[idx]['category']
                }
                results.append(result)
                
                if len(results) >= k:
                    break
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'results': results,
            'search_type': 'exact_nearest_neighbor',
            'embed_type': embed_type,
            'k': k,
            'weights_used': weights,
            'weighted_features': use_weighted_features,
            'enhanced_weights': use_enhanced_weights,
            'total_indexed': len(product_metadata)
        })
    
    except Exception as e:
        return jsonify({'error': f'Exact search error: {str(e)}'}), 500

@app.route('/faiss/search/two-stage', methods=['POST'])
@require_api_key
def faiss_search_two_stage():
    """Two-stage search optimized for 1750 sofa database: coarse retrieval + re-ranking"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        embed_type = 'combined'  # Force combined for best results
        final_k = data.get('k', 6)  # Final results to return (6-8)
        stage1_k = data.get('stage1_k', 300)  # Stage 1: coarse retrieval
        stage2_k = data.get('stage2_k', 50)   # Stage 2: re-ranking pool
        color_filter = data.get('color_filter', None)
        remove_bg = data.get('remove_background', False)  # Background removal option
        use_multiscale = data.get('use_multiscale', False)  # Multi-scale processing option
        scales = data.get('scales', [224, 336, 448])  # Scales for multi-scale processing
        
        if faiss_indexes[embed_type] is None:
            return jsonify({'error': f'No {embed_type} index available'}), 400
        
        # STAGE 1: COARSE RETRIEVAL (1750 → 300)
        # Get query embedding (same as regular search)
        if 'image_url' not in data and 'image' not in data:
            return jsonify({'error': 'Either image_url or image data required'}), 400
        
        image = get_image_from_data(data, remove_bg=remove_bg)
        
        # Get visual embeddings (CLIP + DINOv2)
        if use_multiscale:
            print(f"Using multi-scale processing with scales: {scales}")
            clip_emb, dinov2_emb = process_image_multiscale(image, scales)
        else:
            # Standard single-scale processing
            # CLIP embedding
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_features = clip_model.encode_image(image_input)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_emb = clip_features.cpu().numpy().flatten()
            clip_emb = l2_normalize(clip_emb, "CLIP_query")
            
            # DINOv2 embedding
            dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                dinov2_outputs = dinov2_model(**dinov2_inputs)
                dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
            dinov2_emb = dinov2_emb.cpu().numpy().flatten()
            dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_query")
        
        # Text embedding
        if 'text' in data:
            text = data['text']
        elif 'features' in data:
            text = features_to_text(data['features'], use_weighted=True)
        else:
            return jsonify({'error': 'Text or features required'}), 400
        
        text_emb = get_text_embedding(text)
        if text_emb is None:
            return jsonify({'error': 'Failed to get text embedding'}), 500
        text_emb = l2_normalize(text_emb, "Text_query")
        
        # Combine embeddings for Stage 1 (fast weights)
        stage1_weights = {'clip': 0.35, 'dinov2': 0.25, 'text': 0.40}  # Slightly favor visual for fast filtering
        query_emb_stage1 = combine_embeddings(clip_emb, dinov2_emb, text_emb, stage1_weights)
        query_emb_stage1 = query_emb_stage1.reshape(1, -1).astype('float32')
        
        # Stage 1 search with color filtering
        search_k_stage1 = stage1_k * 2 if color_filter else stage1_k
        distances_s1, indices_s1 = faiss_indexes[embed_type].search(query_emb_stage1, search_k_stage1)
        
        # Filter by color if specified
        stage1_results = []
        for i, (dist, idx) in enumerate(zip(distances_s1[0], indices_s1[0])):
            if len(stage1_results) >= stage1_k:
                break
            
            if idx < len(product_metadata):
                if color_filter:
                    product_color = product_metadata[idx]['features'].get('kolor', '')
                    product_color_group = get_color_group(product_color)
                    if product_color_group != color_filter:
                        continue
                
                stage1_results.append({
                    'distance': float(dist),
                    'index': int(idx),
                    'product': product_metadata[idx]
                })
        
        # STAGE 2: RE-RANKING (300 → 50)
        # Use enhanced weights and metadata scoring
        stage2_weights = EMBEDDING_WEIGHTS_ENHANCED  # [0.25, 0.30, 0.45] - favor text for precise matching
        query_emb_stage2 = combine_embeddings(clip_emb, dinov2_emb, text_emb, stage2_weights)
        
        # Calculate enhanced similarity scores for Stage 1 results
        stage2_scores = []
        for result in stage1_results[:min(len(stage1_results), stage2_k * 2)]:  # Process more for better re-ranking
            idx = result['index']
            
            # Get product embedding for manual similarity calculation
            product_emb = np.array(faiss_indexes[embed_type].reconstruct(idx)).reshape(1, -1)
            
            # Visual similarity (70%)
            visual_sim = cosine_similarity(query_emb_stage2.reshape(1, -1), product_emb)[0][0]
            
            # Category/metadata match (15%)
            category_score = 1.0  # All are sofas, so base score
            product_features = result['product']['features']
            
            # Style match bonus
            if 'features' in data and 'styl' in data['features'] and 'styl' in product_features:
                if data['features']['styl'].lower() == product_features.get('styl', '').lower():
                    category_score += 0.2
            
            # Material match bonus  
            if 'features' in data and 'material' in data['features'] and 'material' in product_features:
                if data['features']['material'].lower() in product_features.get('material', '').lower():
                    category_score += 0.3
            
            # Brand/metadata score (15%) - placeholder for future metadata
            metadata_score = 1.0
            
            # Hybrid scoring: Visual (70%) + Category (15%) + Metadata (15%)
            final_score = visual_sim * 0.70 + category_score * 0.15 + metadata_score * 0.15
            
            stage2_scores.append({
                **result,
                'visual_similarity': float(visual_sim),
                'category_score': float(category_score),
                'metadata_score': float(metadata_score),
                'final_score': float(final_score)
            })
        
        # Sort by final score (HIGHEST first) and take top results
        stage2_scores.sort(key=lambda x: x['final_score'], reverse=True)
        final_results = stage2_scores[:final_k]
        
        # DEBUG: Print sorting verification
        print(f"Stage 2 sorting check - Top 3 final_scores: {[r['final_score'] for r in final_results[:3]]}")
        print(f"Stage 2 sorting check - Top 3 visual_sim: {[r['visual_similarity'] for r in final_results[:3]]}")
        
        return jsonify({
            'results': final_results,
            'search_type': 'two_stage',
            'stage1_retrieved': len(stage1_results),
            'stage2_processed': len(stage2_scores),
            'final_returned': len(final_results),
            'embed_type': embed_type,
            'weights': {
                'stage1': stage1_weights,
                'stage2': stage2_weights
            },
            'color_filter': color_filter,
            'parameters': {
                'final_k': final_k,
                'stage1_k': stage1_k,
                'stage2_k': stage2_k,
                'remove_background': remove_bg,
                'use_multiscale': use_multiscale,
                'scales': scales if use_multiscale else None
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Two-stage search error: {str(e)}'}), 500

@app.route('/faiss/search/enhanced', methods=['POST'])
@require_api_key
def faiss_search_enhanced():
    """Enhanced search with new weights and features"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Force enhanced settings
        data['use_enhanced_weights'] = True
        data['use_weighted_features'] = True
        data['embed_type'] = 'combined'  # Force combined search
        
        # Call the regular search function directly
        from flask import g
        g.enhanced_mode = True  # Flag for enhanced mode
        
        # Use the existing search_faiss function
        return search_faiss()
    
    except Exception as e:
        return jsonify({'error': f'Enhanced search error: {str(e)}'}), 500

@app.route('/preprocess/remove-background', methods=['POST'])
@require_api_key
def preprocess_remove_background():
    """Remove background from image and return processed image"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if not REMBG_AVAILABLE:
            return jsonify({'error': 'Background removal not available. Install rembg package.'}), 500
        
        # Get image
        image = get_image_from_data(data, remove_bg=False)  # Don't auto-remove, we'll do it explicitly
        
        # Remove background
        start_time = time.time()
        processed_image = remove_background(image, use_cache=data.get('use_cache', True))
        processing_time = time.time() - start_time
        
        # Convert back to base64
        output_buffer = io.BytesIO()
        processed_image.save(output_buffer, format='PNG')
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'processed_image': output_base64,
            'processing_time': round(processing_time, 3),
            'cache_used': data.get('use_cache', True),
            'original_size': f"{image.size[0]}x{image.size[1]}",
            'processed_size': f"{processed_image.size[0]}x{processed_image.size[1]}",
            'background_removed': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Background removal error: {str(e)}'}), 500

@app.route('/embeddings/multiscale', methods=['POST'])
@require_api_key
def embeddings_multiscale():
    """Get multi-scale embeddings (CLIP + DINOv2) averaged across different image scales"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Get processing options
        scales = data.get('scales', [224, 336, 448])
        remove_bg = data.get('remove_background', False)
        
        # Get image
        image = get_image_from_data(data, remove_bg=remove_bg)
        
        # Process at multiple scales
        start_time = time.time()
        clip_emb, dinov2_emb = process_image_multiscale(image, scales)
        processing_time = time.time() - start_time
        
        return jsonify({
            'clip_embedding': clip_emb.tolist(),
            'dinov2_embedding': dinov2_emb.tolist(),
            'scales_processed': scales,
            'processing_time': round(processing_time, 3),
            'background_removed': remove_bg,
            'clip_dimensions': len(clip_emb),
            'dinov2_dimensions': len(dinov2_emb),
            'method': 'multiscale_averaging'
        })
        
    except Exception as e:
        return jsonify({'error': f'Multi-scale embedding error: {str(e)}'}), 500

@app.route('/features/test-enhanced', methods=['POST'])
@require_api_key
def test_enhanced_features():
    """Test enhanced product features with new attributes"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Example enhanced features for testing
        sample_features = data.get('features', {
            'material': 'skóra',
            'specyfikacja_materialu': 'skóra naturalna',
            'kolor': 'czarny',
            'styl': 'nowoczesny',
            'podstawa': 'nogi metalowe',
            'typ': 'narożna',
            'rozmiar_cm': '250x180x85',  # New: actual dimensions
            'pojemnosc': '4-osobowa',
            'funkcja_spania': 'tak, rozkładana',  # New: split from cechy_dodatkowe
            'schowek': 'tak, pod siedziskiem',  # New: split from cechy_dodatkowe
            'ksztalt': 'L-shape',
            'pikowana': 'nie',
            'szezlong': 'lewy',
            'kierunek_ustawienia': 'lewy róg',
            'mechanizm': 'rozkładanie automatyczne',  # New: mechanisms
            'cechy_dodatkowe': 'poduszki dekoracyjne'  # Remaining features
        })
        
        # Generate text representations
        text_unweighted = features_to_text(sample_features, use_weighted=False)
        text_weighted = features_to_text(sample_features, use_weighted=True)
        
        # Get embeddings for both
        embedding_unweighted = get_text_embedding(text_unweighted)
        embedding_weighted = get_text_embedding(text_weighted)
        
        # Feature analysis
        feature_analysis = {}
        for feature, weight in FEATURE_WEIGHTS.items():
            value = sample_features.get(feature, '')
            feature_analysis[feature] = {
                'value': value,
                'weight': weight,
                'has_value': bool(value and value != 'null'),
                'is_new_feature': feature in ['rozmiar_cm', 'funkcja_spania', 'schowek', 'mechanizm']
            }
        
        return jsonify({
            'sample_features': sample_features,
            'text_representations': {
                'unweighted': text_unweighted,
                'weighted': text_weighted
            },
            'embeddings': {
                'unweighted_dim': len(embedding_unweighted) if embedding_unweighted else 0,
                'weighted_dim': len(embedding_weighted) if embedding_weighted else 0,
                'unweighted_norm': float(np.linalg.norm(embedding_unweighted)) if embedding_unweighted else 0,
                'weighted_norm': float(np.linalg.norm(embedding_weighted)) if embedding_weighted else 0
            },
            'feature_analysis': feature_analysis,
            'new_features': {
                'rozmiar_cm': 'Actual dimensions in cm (weight: 1.6)',
                'funkcja_spania': 'Sleep function - split from cechy_dodatkowe (weight: 1.1)',
                'schowek': 'Storage function - split from cechy_dodatkowe (weight: 1.0)',
                'mechanizm': 'Mechanisms like reclining, adjustable (weight: 0.7)'
            },
            'total_features': len(FEATURE_WEIGHTS),
            'features_with_values': sum(1 for v in feature_analysis.values() if v['has_value'])
        })
        
    except Exception as e:
        return jsonify({'error': f'Enhanced features test error: {str(e)}'}), 500

@app.route('/admin/cleanup-processed-images', methods=['POST'])
@require_api_key
def cleanup_processed_images_endpoint():
    """Clean up old processed images and metadata"""
    try:
        data = request.get_json()
        force_cleanup = data.get('force', False) if data else False
        max_age_hours = data.get('max_age_hours', 24 * 7) if data else 24 * 7  # Default 7 days
        
        # Override max age if specified
        global PROCESSED_IMAGES_MAX_AGE
        original_max_age = PROCESSED_IMAGES_MAX_AGE
        if data and 'max_age_hours' in data:
            PROCESSED_IMAGES_MAX_AGE = max_age_hours * 3600
        
        # Perform cleanup
        removed_count = cleanup_old_processed_images()
        
        # Get current stats
        total_images = len(processed_images_metadata)
        total_size = 0
        for metadata in processed_images_metadata.values():
            for path in [metadata['original_path'], metadata['processed_path']]:
                if os.path.exists(path):
                    total_size += os.path.getsize(path)
        
        # Restore original max age
        PROCESSED_IMAGES_MAX_AGE = original_max_age
        
        return jsonify({
            'status': 'success',
            'removed_images': removed_count,
            'remaining_images': total_images,
            'total_disk_usage_mb': round(total_size / (1024 * 1024), 2),
            'max_age_used_hours': max_age_hours,
            'force_cleanup': force_cleanup
        })
        
    except Exception as e:
        return jsonify({'error': f'Cleanup error: {str(e)}'}), 500

@app.route('/admin/processed-images-stats', methods=['GET'])
@require_api_key  
def processed_images_stats():
    """Get statistics about processed images storage"""
    try:
        total_images = len(processed_images_metadata)
        total_size = 0
        oldest_timestamp = None
        newest_timestamp = None
        
        for metadata in processed_images_metadata.values():
            # Calculate size
            for path in [metadata['original_path'], metadata['processed_path']]:
                if os.path.exists(path):
                    total_size += os.path.getsize(path)
            
            # Track age
            timestamp = metadata['timestamp']
            if oldest_timestamp is None or timestamp < oldest_timestamp:
                oldest_timestamp = timestamp
            if newest_timestamp is None or timestamp > newest_timestamp:
                newest_timestamp = timestamp
        
        current_time = time.time()
        
        return jsonify({
            'total_processed_images': total_images,
            'total_disk_usage_mb': round(total_size / (1024 * 1024), 2),
            'storage_directory': PROCESSED_IMAGES_DIR,
            'metadata_file': PROCESSED_IMAGES_METADATA_FILE,
            'age_stats': {
                'oldest_hours': round((current_time - oldest_timestamp) / 3600, 1) if oldest_timestamp else 0,
                'newest_hours': round((current_time - newest_timestamp) / 3600, 1) if newest_timestamp else 0,
                'max_age_hours': round(PROCESSED_IMAGES_MAX_AGE / 3600, 1)
            },
            'cache_stats': {
                'memory_cache_size': len(background_removal_cache),
                'memory_cache_max': BG_CACHE_MAX_SIZE,
                'memory_cache_ttl_hours': round(BG_CACHE_TTL / 3600, 1)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Stats error: {str(e)}'}), 500

@app.route('/admin/model-config', methods=['GET'])
@require_api_key
def get_model_config():
    """Get current model configuration and available options"""
    try:
        clip_config = MODEL_CONFIG['clip'][CURRENT_MODEL_SIZE]
        dinov2_config = MODEL_CONFIG['dinov2'][CURRENT_MODEL_SIZE]
        
        return jsonify({
            'current_configuration': {
                'model_size': CURRENT_MODEL_SIZE,
                'clip': {
                    'name': clip_config['name'],
                    'dimensions': clip_config['dim']
                },
                'dinov2': {
                    'name': dinov2_config['name'],
                    'dimensions': dinov2_config['dim']
                },
                'text': {
                    'name': 'text-embedding-3-large',
                    'dimensions': 3072
                }
            },
            'available_configurations': MODEL_CONFIG,
            'combined_dimensions': {
                'small': MODEL_CONFIG['clip']['small']['dim'] + MODEL_CONFIG['dinov2']['small']['dim'] + 3072,
                'large': MODEL_CONFIG['clip']['large']['dim'] + MODEL_CONFIG['dinov2']['large']['dim'] + 3072
            },
            'memory_usage': {
                'gpu_available': torch.cuda.is_available(),
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / 1024**3 if torch.cuda.is_available() else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model config error: {str(e)}'}), 500

@app.route('/admin/switch-model-size', methods=['POST'])
@require_api_key
def switch_model_size():
    """Switch between small and large models (requires restart)"""
    try:
        data = request.get_json()
        if not data or 'model_size' not in data:
            return jsonify({'error': 'model_size parameter required (small or large)'}), 400
        
        new_size = data['model_size']
        if new_size not in ['small', 'large']:
            return jsonify({'error': 'model_size must be "small" or "large"'}), 400
        
        global CURRENT_MODEL_SIZE
        old_size = CURRENT_MODEL_SIZE
        CURRENT_MODEL_SIZE = new_size
        
        # Note: This requires application restart to take effect
        return jsonify({
            'status': 'success',
            'message': f'Model size changed from {old_size} to {new_size}',
            'note': 'Application restart required for changes to take effect',
            'previous_config': {
                'clip': MODEL_CONFIG['clip'][old_size],
                'dinov2': MODEL_CONFIG['dinov2'][old_size]
            },
            'new_config': {
                'clip': MODEL_CONFIG['clip'][new_size],
                'dinov2': MODEL_CONFIG['dinov2'][new_size]
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model switch error: {str(e)}'}), 500

@app.route('/test/model-performance', methods=['POST'])
@require_api_key
def test_model_performance():
    """Test current model performance with sample data"""
    try:
        data = request.get_json()
        test_image_url = data.get('image_url') if data else None
        test_features = data.get('features', {
            'kolor': 'szary',
            'material': 'tkanina',
            'typ': 'sofa',
            'ksztalt': 'narożnik'
        }) if data else {'kolor': 'szary', 'material': 'tkanina', 'typ': 'sofa'}
        
        results = {}
        start_time = time.time()
        
        # Test CLIP model
        if test_image_url:
            try:
                image = load_image_from_url(test_image_url)
                
                clip_start = time.time()
                image_input = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_features = clip_model.encode_image(image_input)
                    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                clip_time = time.time() - clip_start
                
                clip_emb = clip_features.cpu().numpy().flatten()
                clip_emb = l2_normalize(clip_emb, "CLIP_test")
                
                results['clip'] = {
                    'dimensions': len(clip_emb),
                    'processing_time_ms': round(clip_time * 1000, 2),
                    'norm': float(np.linalg.norm(clip_emb)),
                    'sample_values': clip_emb[:5].tolist()
                }
            except Exception as e:
                results['clip'] = {'error': str(e)}
        
        # Test DINOv2 model
        if test_image_url:
            try:
                dinov2_start = time.time()
                dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    dinov2_outputs = dinov2_model(**dinov2_inputs)
                    dinov2_emb = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze()
                dinov2_time = time.time() - dinov2_start
                
                dinov2_emb = dinov2_emb.cpu().numpy().flatten()
                dinov2_emb = l2_normalize(dinov2_emb, "DINOv2_test")
                
                results['dinov2'] = {
                    'dimensions': len(dinov2_emb),
                    'processing_time_ms': round(dinov2_time * 1000, 2),
                    'norm': float(np.linalg.norm(dinov2_emb)),
                    'sample_values': dinov2_emb[:5].tolist()
                }
            except Exception as e:
                results['dinov2'] = {'error': str(e)}
        
        # Test text embedding
        try:
            text_start = time.time()
            text = features_to_text(test_features, use_weighted=True)
            text_emb = get_text_embedding(text)
            text_time = time.time() - text_start
            
            if text_emb is not None:
                text_emb = l2_normalize(text_emb, "Text_test")
                results['text'] = {
                    'dimensions': len(text_emb),
                    'processing_time_ms': round(text_time * 1000, 2),
                    'norm': float(np.linalg.norm(text_emb)),
                    'text_generated': text,
                    'sample_values': text_emb[:5].tolist()
                }
            else:
                results['text'] = {'error': 'Failed to get text embedding'}
        except Exception as e:
            results['text'] = {'error': str(e)}
        
        # Test combined embedding
        if 'clip' in results and 'dinov2' in results and 'text' in results:
            if not any('error' in r for r in results.values()):
                try:
                    combined_start = time.time()
                    clip_emb = clip_features.cpu().numpy().flatten()
                    clip_emb = l2_normalize(clip_emb, "CLIP_combine")
                    dinov2_emb_combined = dinov2_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().flatten()
                    dinov2_emb_combined = l2_normalize(dinov2_emb_combined, "DINOv2_combine")
                    
                    combined_emb = combine_embeddings(clip_emb, dinov2_emb_combined, text_emb, EMBEDDING_WEIGHTS_ENHANCED)
                    combined_time = time.time() - combined_start
                    
                    results['combined'] = {
                        'dimensions': len(combined_emb),
                        'processing_time_ms': round(combined_time * 1000, 2),
                        'norm': float(np.linalg.norm(combined_emb)),
                        'weights_used': EMBEDDING_WEIGHTS_ENHANCED,
                        'sample_values': combined_emb[:5].tolist()
                    }
                except Exception as e:
                    results['combined'] = {'error': str(e)}
        
        total_time = time.time() - start_time
        
        return jsonify({
            'model_config': {
                'size': CURRENT_MODEL_SIZE,
                'clip': MODEL_CONFIG['clip'][CURRENT_MODEL_SIZE],
                'dinov2': MODEL_CONFIG['dinov2'][CURRENT_MODEL_SIZE],
                'text': {'name': 'text-embedding-3-large', 'dim': 3072}
            },
            'test_results': results,
            'total_time_ms': round(total_time * 1000, 2),
            'gpu_memory': {
                'allocated_gb': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
                'reserved_gb': torch.cuda.memory_reserved(0) / 1024**3 if torch.cuda.is_available() else 0
            },
            'test_inputs': {
                'image_url': test_image_url,
                'features': test_features
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model performance test error: {str(e)}'}), 500

@app.route('/preprocess/batch-remove-background', methods=['POST'])
@require_api_key
def batch_remove_background():
    """Process multiple images for background removal in batch"""
    try:
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({'error': 'images array required'}), 400
        
        images_data = data['images']
        use_cache = data.get('use_cache', True)
        save_to_disk = data.get('save_to_disk', True)
        
        if not REMBG_AVAILABLE:
            return jsonify({'error': 'Background removal not available. Install rembg package.'}), 500
        
        if len(images_data) > 50:
            return jsonify({'error': 'Maximum 50 images per batch'}), 400
        
        results = []
        total_start_time = time.time()
        cache_hits = 0
        processing_count = 0
        
        for i, image_data in enumerate(images_data):
            try:
                # Get image
                image = get_image_from_data(image_data, remove_bg=False)
                
                # Process with background removal
                start_time = time.time()
                processed_image = remove_background(image, use_cache=use_cache, save_to_disk=save_to_disk)
                processing_time = time.time() - start_time
                
                # Check if it was a cache hit (very fast processing)
                if processing_time < 0.1:
                    cache_hits += 1
                else:
                    processing_count += 1
                
                # Convert back to base64 if requested
                output_base64 = None
                if data.get('return_images', False):
                    output_buffer = io.BytesIO()
                    processed_image.save(output_buffer, format='PNG')
                    output_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                
                results.append({
                    'index': i,
                    'status': 'success',
                    'processing_time': round(processing_time, 3),
                    'cache_hit': processing_time < 0.1,
                    'original_size': f"{image.size[0]}x{image.size[1]}",
                    'processed_size': f"{processed_image.size[0]}x{processed_image.size[1]}",
                    'processed_image': output_base64
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'status': 'error',
                    'error': str(e)
                })
        
        total_time = time.time() - total_start_time
        
        return jsonify({
            'status': 'completed',
            'total_images': len(images_data),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'error']),
            'cache_hits': cache_hits,
            'processed_count': processing_count,
            'total_time': round(total_time, 2),
            'avg_time_per_image': round(total_time / len(images_data), 3),
            'results': results,
            'cache_statistics': {
                'memory_cache_size': len(background_removal_cache),
                'persistent_storage_size': len(processed_images_metadata)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch background removal error: {str(e)}'}), 500

@app.route('/', methods=['GET'])
@public_endpoint
def home():
    return jsonify({
        'message': 'Flask Embedding Server - Furniture Similarity Search',
        'endpoints': {
            '/clip': 'POST - Get CLIP embeddings from base64 image (L2 normalized)',
            '/dino': 'POST - Get DINOv2 embeddings from base64 image (L2 normalized)',
            '/text': 'POST - Get text embeddings from features or text (L2 normalized)',
            '/combined': 'POST - Get combined embeddings (CLIP + DINOv2 + Text, L2 normalized)',
            '/faiss/build': 'POST - Build FAISS index from products (all embeddings L2 normalized)',
            '/faiss/add': 'POST - Add products to existing FAISS index (batch processing)',
            '/faiss/search': 'POST - Search similar products using FAISS (query normalized)',
            '/faiss/search/two-stage': 'POST - Two-stage search: coarse retrieval (1750→300) + re-ranking (300→6-8)',
            '/faiss/search/enhanced': 'POST - Enhanced search with optimized weights',
            '/faiss/search/exact': 'POST - Exact nearest neighbor search (brute force)',
            '/preprocess/remove-background': 'POST - Remove background from image using Rembg',
            '/embeddings/multiscale': 'POST - Multi-scale image embeddings (CLIP + DINOv2)',
            '/features/test-enhanced': 'POST - Test enhanced product features with new attributes',
            '/admin/cleanup-processed-images': 'POST - Clean up old processed images and metadata',
            '/admin/processed-images-stats': 'GET - Get statistics about processed images storage',
            '/admin/model-config': 'GET - Get current model configuration and available options',
            '/admin/switch-model-size': 'POST - Switch between small and large models (requires restart)',
            '/test/model-performance': 'POST - Test current model performance with sample data',
            '/preprocess/batch-remove-background': 'POST - Process multiple images for background removal',
            '/faiss/stats': 'GET - Get FAISS index statistics',
            '/test/rnn': 'POST - Test Reciprocal Nearest Neighbors',
            '/test/rnn-enhanced': 'POST - Enhanced RNN test with exact NN and threshold config',
            '/batch': 'POST - Batch process multiple products (all normalized)',
            '/debug/verify-norm': 'POST - Verify L2 normalization of embeddings',
            '/health': 'GET - Health check'
        },
        'features': {
            'gpu_acceleration': torch.cuda.is_available(),
            'current_model_size': CURRENT_MODEL_SIZE,
            'models': {
                'clip': MODEL_CONFIG['clip'][CURRENT_MODEL_SIZE],
                'dinov2': MODEL_CONFIG['dinov2'][CURRENT_MODEL_SIZE],
                'text': {'name': 'text-embedding-3-large', 'dim': 3072}
            },
            'embedding_weights': EMBEDDING_WEIGHTS,
            'faiss_indexes': list(faiss_indexes.keys()),
            'products_indexed': len(product_metadata),
            'processed_images_stored': len(processed_images_metadata),
            'l2_normalization': 'All embeddings normalized to unit norm (L2 = 1.0)',
            'normalization_epsilon': 1e-6,
            'security': {
                'api_key_required': True,
                'rate_limiting': f'{MAX_REQUESTS_PER_MINUTE} requests per minute per IP',
                'public_endpoints': ['/', '/health', '/faiss/stats']
            }
        },
        'usage': {
            'authentication': 'Add header: X-API-Key: szuk_ai_embeddings_2024_secure_key',
            'basic_embedding': 'POST with JSON: {"image": "base64_data"}',
            'text_embedding': 'POST with JSON: {"features": {...}} or {"text": "..."}',
            'combined_embedding': 'POST with JSON: {"image": "base64_data", "features": {...}}',
            'similarity_search': 'POST with JSON: {"embed_type": "combined", "image": "...", "k": 10}',
            'build_index': 'POST with JSON: {"products": [{"id": 1, "image": "...", "features": {...}}]}',
            'add_to_index': 'POST with JSON: {"products": [...], "append": true}',
            'verify_normalization': 'POST with JSON: {"image": "base64_data", "features": {...}}'
        },
        'api_key': API_KEY
    })

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)