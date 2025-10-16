"""
Data utility functions for Amazon Reviews project
Functions for loading, processing, and preparing datasets
"""

from collections import defaultdict
import pandas as pd
import numpy as np


def build_pairs(names):
    """
    Build dataset pairs from list of dataset names.
    Pairs review and meta datasets for each category.
    
    Args:
        names: List of dataset names
        
    Returns:
        Dictionary mapping categories to their review/meta dataset names
    """
    pairs = defaultdict(dict)
    for name in names:
        if name.startswith("raw_review_"):
            cat = name.replace("raw_review_", "")
            pairs[cat]["review"] = name
        elif name.startswith("raw_meta_"):
            cat = name.replace("raw_meta_", "")
            pairs[cat]["meta"] = name
    return pairs


def load_review_and_meta(review_name, meta_name):
    """
    Load review and metadata datasets from HuggingFace.
    
    Args:
        review_name: Name of the review dataset
        meta_name: Name of the metadata dataset
        
    Returns:
        Tuple of (df_review, df_meta) DataFrames
    """
    from datasets import load_dataset
    dataset_review = load_dataset("McAuley-Lab/Amazon-Reviews-2023", review_name, split="full", trust_remote_code=True)
    dataset_meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", meta_name, split="full", trust_remote_code=True)
    df_review = dataset_review.to_pandas()
    df_meta = dataset_meta.to_pandas()
    return df_review, df_meta


def preprocess_df(df_review, df_meta):
    """
    Preprocess and merge review and metadata DataFrames.
    
    Args:
        df_review: Review DataFrame
        df_meta: Metadata DataFrame
        
    Returns:
        Merged and cleaned DataFrame
    """
    df_meta = df_meta[['parent_asin','main_category','title','average_rating','rating_number','features','description','price','images','categories','store']]
    df_review = df_review[['asin','parent_asin','user_id','rating','title','text','timestamp','helpful_vote','verified_purchase','images']]
    df_meta = df_meta.dropna(subset = ['parent_asin']).drop_duplicates(subset = ['parent_asin'])
    df = df_review.merge(df_meta, on= 'parent_asin', how='left')
    return df


def to_text(x):
    """
    Convert scalars/lists to a single clean string.
    
    Args:
        x: Input value (can be None, list, or scalar)
        
    Returns:
        Cleaned string representation
    """
    if x is None:
        return ""
    if isinstance(x, list):
        # description often is a list of strings
        return " ".join([str(t) for t in x if t is not None])
    return str(x)


def prepare_text_columns(df):
    """
    Prepare text columns for embedding generation.
    Creates review_text and meta_text columns from existing columns.
    
    Args:
        df: DataFrame with review and meta data
        
    Returns:
        DataFrame with added text columns
    """
    # Review-side text: user-written title_x + review body text
    df["review_text"] = (
        df.get("title_x", "").apply(to_text).fillna("") + " " +
        df.get("text", "").apply(to_text).fillna("")
    ).str.strip()

    # Meta-side text: product title_y + product description
    df["meta_text"] = (
        df.get("title_y", "").apply(to_text).fillna("") + " " +
        df.get("description", "").apply(to_text).fillna("")
    ).str.strip()
    
    return df
