"""
FAISS utility functions
Functions for building and querying FAISS indexes for similarity search
"""

import numpy as np
import faiss
from sklearn.preprocessing import normalize


def build_faiss_index(embeddings, index_type="flat", normalize_emb=True):
    """
    Build a FAISS index from embeddings.
    
    Args:
        embeddings: Embedding matrix (numpy array)
        index_type: Type of index ("flat", "ivfflat", "hnsw")
        normalize_emb: Whether to normalize embeddings before indexing
        
    Returns:
        FAISS index
    """
    embeddings = embeddings.astype('float32')
    
    if normalize_emb:
        embeddings = normalize(embeddings, axis=1)
    
    d = embeddings.shape[1]  # Dimension
    
    if index_type == "flat":
        # Flat index with inner product (for cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(d)
    elif index_type == "ivfflat":
        # IVF index for larger datasets
        nlist = min(100, len(embeddings) // 100)  # Number of clusters
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        # Need to train IVF index
        index.train(embeddings)
    elif index_type == "hnsw":
        # HNSW index for fast approximate search
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add vectors to index
    index.add(embeddings)
    
    return index


def search_faiss_index(index, query_vectors, k=10, normalize_query=True):
    """
    Search FAISS index for nearest neighbors.
    
    Args:
        index: FAISS index
        query_vectors: Query vector(s) to search for
        k: Number of nearest neighbors
        normalize_query: Whether to normalize query vectors
        
    Returns:
        Tuple of (similarities, indices)
    """
    # Ensure query is 2D array
    if len(query_vectors.shape) == 1:
        query_vectors = query_vectors.reshape(1, -1)
    
    query_vectors = query_vectors.astype('float32')
    
    if normalize_query:
        query_vectors = normalize(query_vectors, axis=1)
    
    # Search
    similarities, indices = index.search(query_vectors, k)
    
    return similarities, indices


def create_multimodal_query(text_query, text_model, text_dim=384, img_dim=512, 
                           text_weight=1.0, normalize=True):
    """
    Create a multimodal query vector from text input.
    
    Args:
        text_query: Text query string
        text_model: SentenceTransformer model
        text_dim: Dimension of text embeddings
        img_dim: Dimension of image embeddings
        text_weight: Weight for text component (0 to 1)
        normalize: Whether to normalize the final query
        
    Returns:
        Query vector with concatenated text and zero image components
    """
    # Encode text query
    text_emb = text_model.encode([text_query])[0]
    
    # Normalize text embedding
    text_emb = text_emb / np.linalg.norm(text_emb)
    
    # Create zero padding for image component
    img_padding = np.zeros(img_dim)
    
    # Concatenate with optional weighting
    if text_weight != 1.0:
        text_emb = text_emb * text_weight
    
    query = np.concatenate([text_emb, img_padding], axis=0)
    
    # Normalize final query
    if normalize:
        query = query / np.linalg.norm(query)
    
    return query


def save_faiss_index(index, filepath):
    """
    Save FAISS index to disk.
    
    Args:
        index: FAISS index
        filepath: Path to save the index
    """
    faiss.write_index(index, filepath)


def load_faiss_index(filepath):
    """
    Load FAISS index from disk.
    
    Args:
        filepath: Path to the saved index
        
    Returns:
        FAISS index
    """
    return faiss.read_index(filepath)


def get_search_results(index, query, merged_df, k=10, columns_to_show=None):
    """
    Get formatted search results from FAISS index.
    
    Args:
        index: FAISS index
        query: Query vector
        merged_df: DataFrame with metadata for results
        k: Number of results
        columns_to_show: List of columns to include in results
        
    Returns:
        DataFrame with search results
    """
    similarities, indices = search_faiss_index(index, query, k=k)
    
    # Get results from dataframe
    results = merged_df.iloc[indices[0]].copy()
    results['similarity_score'] = similarities[0]
    
    if columns_to_show:
        columns = columns_to_show + ['similarity_score']
        results = results[columns]
    
    return results