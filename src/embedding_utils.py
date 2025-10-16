"""
Embedding utility functions
Functions for text embedding, fusion, and similarity search
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


def get_text_model(model_name="all-MiniLM-L6-v2"):
    """
    Load and return a text embedding model.
    
    Args:
        model_name: Name of the SentenceTransformer model
        
    Returns:
        SentenceTransformer model
    """
    return SentenceTransformer(model_name)


def generate_text_embeddings(texts, model, batch_size=64):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        
    Returns:
        Numpy array of embeddings
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")


def fuse_embeddings(emb1, emb2, alpha=0.7, normalize_output=True):
    """
    Fuse two embedding matrices with weighted combination.
    
    Args:
        emb1: First embedding matrix
        emb2: Second embedding matrix
        alpha: Weight for first embedding (1-alpha for second)
        normalize_output: Whether to normalize the fused embeddings
        
    Returns:
        Fused embedding matrix
    """
    fused = alpha * emb1 + (1 - alpha) * emb2
    
    if normalize_output:
        fused = normalize(fused, axis=1)
    
    return fused


def prepare_multimodal_embeddings(text_emb, img_emb, normalize_separately=True):
    """
    Prepare multimodal embeddings by concatenating text and image embeddings.
    
    Args:
        text_emb: Text embeddings
        img_emb: Image embeddings
        normalize_separately: Whether to normalize each modality before concatenation
        
    Returns:
        Concatenated and normalized embeddings
    """
    if normalize_separately:
        text_emb = normalize(text_emb, axis=1)
        img_emb = normalize(img_emb, axis=1)
    
    # Concatenate
    multimodal_emb = np.concatenate([text_emb, img_emb], axis=1)
    
    # Normalize the concatenated embeddings
    return normalize(multimodal_emb, axis=1)


def find_nearest_neighbors(query_emb, index_emb, k=10, metric="cosine"):
    """
    Find k nearest neighbors using sklearn.
    
    Args:
        query_emb: Query embedding(s)
        index_emb: Index embeddings to search
        k: Number of neighbors
        metric: Distance metric
        
    Returns:
        Indices and distances of nearest neighbors
    """
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(index_emb)
    
    if len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
    
    distances, indices = nbrs.kneighbors(query_emb)
    
    if metric == "cosine":
        # Convert cosine distance to similarity
        similarities = 1 - distances
        return indices, similarities
    
    return indices, distances