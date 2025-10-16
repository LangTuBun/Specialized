"""
CLIP image processing functions
Functions for downloading, processing, and embedding images using CLIP
"""

import os
import io
import requests
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def get_clip_model(device=None):
    """
    Load and initialize CLIP model.
    
    Args:
        device: Device to run model on (None for auto-detect)
        
    Returns:
        Tuple of (model, processor, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device


def get_session():
    """
    Create a requests session with proper headers.
    
    Returns:
        Configured requests.Session
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
        )
    })
    return session


def fetch_image_bytes(url, session=None, timeout=2.5, max_kb=4096, fname=None, cache_dir=None):
    """
    Download image or load from cache.
    
    Args:
        url: Image URL to fetch
        session: Requests session to use
        timeout: Request timeout in seconds
        max_kb: Maximum file size in KB
        fname: Filename for caching
        cache_dir: Directory for caching images
        
    Returns:
        Image bytes or None if failed
    """
    if session is None:
        session = get_session()
        
    path = os.path.join(cache_dir, fname + ".jpg") if fname and cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    try:
        r = session.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        b = r.content
        if len(b) > max_kb * 1024:
            return None
        if path:
            with open(path, "wb") as f:
                f.write(b)
        return b
    except Exception:
        return None


def to_pil(b):
    """
    Convert bytes to PIL Image.
    
    Args:
        b: Image bytes
        
    Returns:
        PIL Image or None if conversion fails
    """
    if b is None:
        return None
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None


@torch.no_grad()
def embed_pil_batch(pil_list, processor, model, device, batch_size=32):
    """
    Embed a list of PIL images in batches.
    
    Args:
        pil_list: List of PIL images (can contain None values)
        processor: CLIP processor
        model: CLIP model
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        List of embeddings (None for failed images)
    """
    keep_idx, images = [], []
    for i, im in enumerate(pil_list):
        if im is not None:
            keep_idx.append(i)
            images.append(im)
    
    embs = [None] * len(pil_list)
    for s in range(0, len(images), batch_size):
        batch = images[s:s + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = feats.half()  # halve VRAM
        feats = torch.nn.functional.normalize(feats, p=2, dim=1).cpu().numpy().astype("float16")
        for j, vec in enumerate(feats):
            embs[keep_idx[s + j]] = vec
        del inputs, feats
        torch.cuda.empty_cache()
    return embs


def pick_best_image_from_images_field(images_field):
    """
    Returns a single best URL given one row's images field.
    Prefers hi_res MAIN -> hi_res any -> large MAIN -> large any -> thumb MAIN -> thumb any.
    
    Args:
        images_field: Images field from dataset (dict, list, or None)
        
    Returns:
        Best image URL or None if nothing usable
    """
    if images_field is None:
        return None

    # Some splits may store images as a plain list of URLs (reviews_x). Handle that too.
    if isinstance(images_field, (list, tuple)):
        return images_field[0] if images_field else None

    # Expect dict with arrays
    if isinstance(images_field, dict):
        # Normalize to python lists
        def to_list(x):
            if x is None: return []
            if isinstance(x, np.ndarray): return x.tolist()
            if isinstance(x, (list, tuple)): return list(x)
            return []

        hi_res   = to_list(images_field.get('hi_res'))
        large    = to_list(images_field.get('large'))
        thumb    = to_list(images_field.get('thumb'))
        variant  = to_list(images_field.get('variant'))

        # If no variant provided, just prefer hi_res->large->thumb by first URL
        if not variant:
            for arr in (hi_res, large, thumb):
                if arr: return arr[0]
            return None

        # Build aligned records
        L = max(len(hi_res), len(large), len(thumb), len(variant))
        recs = []
        for i in range(L):
            v = variant[i] if i < len(variant) else None
            recs.append({
                'variant': v,
                'hi_res': hi_res[i] if i < len(hi_res) else None,
                'large':  large[i]  if i < len(large)  else None,
                'thumb':  thumb[i]  if i < len(thumb)  else None,
            })

        # Priority helpers
        def first_where(key, cond=lambda r: True):
            for r in recs:
                if cond(r) and r.get(key): return r[key]
            return None

        # Try in order of preference
        return ( first_where('hi_res',  lambda r: r['variant'] == 'MAIN')
              or first_where('hi_res')
              or first_where('large',   lambda r: r['variant'] == 'MAIN')
              or first_where('large')
              or first_where('thumb',   lambda r: r['variant'] == 'MAIN')
              or first_where('thumb') )

    # Fallback: unknown type
    return None
