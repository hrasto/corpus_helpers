import itertools
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE


# --- topic model ---

def fit_topic_model(docs_vect, model_class, **model_kwargs):
    """Fit a topic model on a document-term matrix. Returns the fitted model."""
    return model_class(**model_kwargs).fit(docs_vect)


def save_topic_model(model, vectorizer, path):
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{path}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


def load_topic_model(path):
    with open(f'{path}/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{path}/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# keep old name as alias
load_tm = load_topic_model


# --- plotting ---

def plot_top_words(model, feature_names, n_top_words=15, title=''):
    """Bar chart of top words for each topic component. Returns (fig, axes)."""
    import matplotlib.pyplot as plt
    import math

    n = len(model.components_)
    grid_side = math.ceil(math.sqrt(n))
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(grid_side * 5, grid_side * 5), sharex=True)
    # TODO: plt.subplots(1, 1) returns a bare Axes, not an array — flatten() crashes for n=1
    axes = np.array(axes).flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_ind = topic.argsort()[-n_top_words:]
        top_features = np.array(feature_names)[top_ind]
        weights = topic[top_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 30})
        ax.tick_params(axis='both', which='major', labelsize=20)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
    # TODO: hide unused axes when n_topics < grid_side²
    if title:
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return fig, axes


def fit_tsne(docs_latent, n_components=2, n_points=None, seed=0):
    """Fit t-SNE on docs_latent, optionally subsampling to n_points. Returns 2D array."""
    # TODO: sequential slice may be biased if docs are ordered by corpus; consider shuffling first
    X = docs_latent[:n_points] if n_points is not None else docs_latent
    return TSNE(n_components=n_components, random_state=seed).fit_transform(X)


def plot_tsne(coords_2d, color_by, ax=None, **scatter_kwargs):
    """Scatter plot of 2D t-SNE coords coloured by labels. Returns ax."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    labels = np.array(color_by)
    unique = sorted(set(labels))
    for label in unique:
        mask = labels == label
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], label=label, **scatter_kwargs)
    return ax


# --- partitioning ---

def partition(docs_latent, n_clusters, seed):
    """KMeans clustering on latent representations. Returns integer assignment array."""
    return KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(docs_latent)


def get_region_sizes(assign, file_sizes):
    """
    Compute total size per region.

    assign: array-like of region ids, one per document
    file_sizes: array-like of sizes (bytes), one per document

    Returns dict {region_id: size_in_MB}.
    """
    assign = np.asarray(assign)
    file_sizes = np.asarray(file_sizes, dtype=float)
    regions = {}
    for r in np.unique(assign):
        regions[int(r)] = float(file_sizes[assign == r].sum() / 1e6)
    return regions


def split_largest_region(assign, docs_latent, region_sizes, seed):
    """
    Split the largest region (by size) into two with KMeans(n_clusters=2).
    One half keeps the original region id; the other gets max(assign)+1.
    Returns a new assignment array (does not modify in place).

    region_sizes: dict {region_id: size} as returned by get_region_sizes()
    """
    assign = np.array(assign, copy=True)
    largest = max(region_sizes, key=region_sizes.__getitem__)
    indices = np.where(assign == largest)[0]
    sub_assign = KMeans(n_clusters=2, random_state=seed).fit_predict(docs_latent[indices])
    new_id = int(assign.max()) + 1
    sub_assign = sub_assign.astype(assign.dtype)
    assign[indices] = np.where(sub_assign == 0, largest, new_id)
    return assign


def make_subsets(assign, docs_latent, subset_size, metric='cosine'):
    """
    Find the most similar and most dissimilar subsets of regions of a given size.

    Computes mean pairwise distance between region centroids for every combination
    of `subset_size` regions, then returns (lowest_idx, highest_idx).
    """
    assign = np.asarray(assign)
    region_ids = sorted(np.unique(assign).tolist())
    centroids = np.array([docs_latent[assign == r].mean(axis=0) for r in region_ids])
    dist = pairwise_distances(centroids, metric=metric)

    lowest_idx, lowest_score = None, float('inf')
    highest_idx, highest_score = None, float('-inf')

    # TODO: O(C(n_regions, subset_size)) — warn or bail out for large inputs
    for idx in itertools.combinations(range(len(region_ids)), subset_size):
        # TODO: diagonal (self-distance = 0) inflates the mean; consider masking it out
        score = dist[np.ix_(idx, idx)].mean()
        if score < lowest_score:
            lowest_score = score
            lowest_idx = idx
        if score > highest_score:
            highest_score = score
            highest_idx = idx

    return (
        tuple(region_ids[i] for i in lowest_idx),
        tuple(region_ids[i] for i in highest_idx),
    )
