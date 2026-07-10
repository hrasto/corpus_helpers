import itertools
import os
import pickle
import logging
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from k_means_constrained import KMeansConstrained


def _clamp(value, min_value=0, max_value=1):
    return max(min(value, max_value), min_value)


class KMeansConstrainedRel(KMeansConstrained):
    def __init__(self, n_clusters=8, size_min=None, size_max=None, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=False, random_state=None, copy_x=True, n_jobs=1):
        """ size_min, size_max: if None, will be determined as .5/n_clusters and 2/n_clusters """
        self._size_min = size_min
        self._size_max = size_max
        super().__init__(n_clusters, size_min, size_max, init, n_init, max_iter, tol, verbose, random_state, copy_x, n_jobs)

    def fit_predict(self, X, y=None):
        n_docs = X.shape[0]

        if isinstance(self._size_min, float):
            self.size_min = int(n_docs * _clamp(self._size_min))
        elif self._size_min is None:
            self.size_min = int(n_docs * .5 / self.n_clusters)
        elif isinstance(self._size_min, int):
            self.size_min = self._size_min
        else:
            raise ValueError(f"invalid size_min type: received {type(self.size_min)}, expected float, int, or None")

        if isinstance(self._size_max, float):
            self.size_max = int(n_docs * _clamp(self._size_max))
        elif self._size_max is None:
            self.size_max = int(n_docs * 2 / self.n_clusters)
        elif isinstance(self._size_max, int):
            self.size_max = self._size_max
        else:
            raise ValueError(f"invalid size_min type: received {type(self.size_max)}, expected float, int, or None")

        return super().fit_predict(X, y)


# --- topic model ---

def fit_topic_model(docs: Iterable[Iterable[str]], model_class = LatentDirichletAllocation, model_kwargs:dict=None, cv_kwargs:dict=None):
    """Fit a topic model on a document-term matrix. Returns the fitted model."""
    cv_kwargs = cv_kwargs or {}
    cv = CountVectorizer(**cv_kwargs)
    docs_vect = cv.fit_transform(docs)
    model_kwargs = model_kwargs or {}
    topic_model = model_class(**model_kwargs)
    docs_latent = topic_model.fit_transform(docs_vect)
    return docs_latent, (topic_model, cv)

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


def fit_transform_tsne(docs_latent, n_components=2, n_points=None, seed=0):
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

def partition(docs_latent, n_clusters, clusterer=None):
    """
    Cluster docs_latent using an sklearn clusterer class.

    n_clusters: int → flat partition, returns 1-D integer assignment array.
    n_clusters: list[int] → hierarchical partition, returns 2-D array of shape
        (n_docs, n_levels).  Column i holds the local cluster id at level i
        (0 … n_clusters[i]-1 within each parent group); the full row is the
        path that uniquely identifies a leaf cluster.

    clusterer: sklearn clustering class instantiated as
        clusterer(n_clusters=k, random_state=seed) at each step.
        KMeans (default), BisectingKMeans, and SpectralClustering are
        all compatible.
    """
    if clusterer is None: 
        clusterer = KMeans
        logging.warning(f"running KMeans without setting the random state")
    
    def _cluster(X, k):
        return clusterer(n_clusters=k).fit_predict(X)

    if isinstance(n_clusters, int):
        return _cluster(docs_latent, n_clusters)

    n_docs = len(docs_latent)
    assignments = np.zeros((n_docs, len(n_clusters)), dtype=int)

    assignments[:, 0] = _cluster(docs_latent, n_clusters[0])

    for level in range(1, len(n_clusters)):
        k = n_clusters[level]
        for parent_id in np.unique(assignments[:, level - 1]):
            mask = assignments[:, level - 1] == parent_id
            sub = docs_latent[mask]
            if len(sub) <= k:
                assignments[mask, level] = np.arange(len(sub))
            else:
                assignments[mask, level] = _cluster(sub, k)

    return assignments


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
