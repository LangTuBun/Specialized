"""
Visualization utility functions
Functions for t-SNE, clustering, and plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_tsne(embeddings, n_components=2, perplexity=30, metric="cosine", random_state=42):
    """
    Compute t-SNE projection of embeddings.
    
    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of dimensions for projection
        perplexity: t-SNE perplexity parameter
        metric: Distance metric
        random_state: Random seed
        
    Returns:
        Low-dimensional projection
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(X_2d, labels=None, figsize=(10, 8), title="t-SNE Visualization"):
    """
    Plot t-SNE visualization.
    
    Args:
        X_2d: 2D t-SNE projection
        labels: Optional labels for coloring
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    if labels is not None:
        # Convert labels to numeric codes for coloring
        label_codes = pd.factorize(labels)[0]
        plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                   c=label_codes, cmap="tab20", s=5, alpha=0.7)
        plt.colorbar()
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.7)
    
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    return plt.gcf()


def elbow_method(embeddings, k_range=range(2, 21), random_state=42):
    """
    Perform elbow method for finding optimal number of clusters.
    
    Args:
        embeddings: Data to cluster
        k_range: Range of K values to try
        random_state: Random seed
        
    Returns:
        Dictionary with K values, inertias, and silhouette scores
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score
        if k > 1 and k < len(embeddings):
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(None)
    
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


def plot_elbow_method(elbow_results, figsize=(12, 5)):
    """
    Plot elbow method results.
    
    Args:
        elbow_results: Results from elbow_method function
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot inertia
    ax1.plot(elbow_results['k_values'], elbow_results['inertias'], 'bo-')
    ax1.set_xlabel('Number of clusters (K)')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax1.set_title('Elbow Method for Optimal K')
    ax1.grid(True, alpha=0.3)
    
    # Plot silhouette score
    valid_scores = [(k, s) for k, s in zip(elbow_results['k_values'], 
                                           elbow_results['silhouette_scores']) 
                    if s is not None]
    if valid_scores:
        k_vals, scores = zip(*valid_scores)
        ax2.plot(k_vals, scores, 'ro-')
        ax2.set_xlabel('Number of clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs K')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def perform_clustering(embeddings, n_clusters=6, random_state=42):
    """
    Perform K-means clustering on embeddings.
    
    Args:
        embeddings: Data to cluster
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Cluster labels and KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def plot_clusters(X_2d, cluster_labels, figsize=(8, 6), title="Clustering Results"):
    """
    Plot clustering results on 2D projection.
    
    Args:
        X_2d: 2D projection of data
        cluster_labels: Cluster assignments
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=figsize)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="tab10", s=5)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(label="Cluster")
    return plt.gcf()