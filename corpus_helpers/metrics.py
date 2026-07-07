"""Domain-distance metrics between two text corpora.

Each corpus is represented as an iterable of byte strings.
Important: assumes corpora are pre-tokenized at the word level.

BPE metrics additionally require the `tokenizers` package (HuggingFace).
"""

from __future__ import annotations

import json
import logging
import random
import re
import zlib
from collections import Counter
from typing import Callable, Iterable, Literal
from .tokenizers2 import BPETokenizer, BPEFromMerges
from .read import lower, split_word, split_line, delete_blank, delete_newline, chain_preprocessors
import numpy as np

Unit = Literal["char", "word"]

_log = logging.getLogger(__name__)

# Texts shorter than this (in bytes) may produce unreliable NCD values
# because the compressor's fixed-overhead bytes dominate the compressed size.
_NCD_SHORT_TEXT_THRESHOLD = 100

# letters | digit runs | punctuation/symbols — never mixes categories across boundaries
# _WORD_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+", re.UNICODE)


# def _word_tokenize(text: str) -> list[str]:
    # return _WORD_RE.findall(text)


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    """ "everygrams" up to n """
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            yield tuple(tokens[i : i + k])

def _ngram_counts(corpus: Iterable[bytes], n: int = 1, unit: Unit = "word") -> Counter:
    """Count n-grams over a corpus of documents. Decodes corpus into utf-8. Note: word-mode simply takes items of corpus as words, char mode the (decoded) elements after calling list() on every item of corpus. 

    unit="char" splits each document into characters before forming n-grams;
    unit="word" splits on whitespace.
    """
    counts: Counter = Counter()
    if unit == 'char': 
        for doc in corpus: 
            tokens = list(doc.decode("utf-8", errors="replace"))
            counts.update(_ngrams(tokens, n))
    elif unit == 'word': 
        tokens = [word.decode("utf-8", errors="replace") for word in corpus]
        counts.update(_ngrams(tokens, n))
    
    return counts


def _to_prob(counts: Counter) -> dict:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("empty corpus: cannot form a probability distribution")
    return {k: v / total for k, v in counts.items()}


def ngram_overlap(
    corpus_a: Iterable[bytes],
    corpus_b: Iterable[bytes],
    n: int = 1,
    unit: Unit = "word",
) -> float:
    """Jaccard overlap between the n-gram vocabularies of two corpora."""
    counts_a = _ngram_counts(corpus_a, n=n, unit=unit)
    counts_b = _ngram_counts(corpus_b, n=n, unit=unit)
    keys_a, keys_b = set(counts_a), set(counts_b)
    union = keys_a | keys_b
    if not union:
        return 0.0
    return len(keys_a & keys_b) / len(union)


def ngram_divergence(
    corpus_a: Iterable[bytes],
    corpus_b: Iterable[bytes],
    method: Literal["jsd", "kld"] = "jsd",
    smoothing: float = 1.0,
    n: int = 1,
    unit: Unit = "word",
) -> float:
    """Kullback-Leibler or Jensen-Shannon divergence between two n-gram frequency distributions.

    KL divergence is computed over the union vocabulary, smoothed with additive
    (Laplace) smoothing to avoid undefined log(0/x) terms.

    `smoothing` sets the pseudocount added to every vocabulary entry before
    normalisation (default 1.0 = add-one smoothing).
    """
    counts_a = _ngram_counts(corpus_a, n=n, unit=unit)
    counts_b = _ngram_counts(corpus_b, n=n, unit=unit)
    vocab = set(counts_a) | set(counts_b)
    if not vocab:
        raise ValueError("both count distributions are empty")
    total_a = sum(counts_a.values()) + smoothing * len(vocab)
    total_b = sum(counts_b.values()) + smoothing * len(vocab)
    p = np.array([(counts_a.get(k, 0) + smoothing) / total_a for k in vocab])
    q = np.array([(counts_b.get(k, 0) + smoothing) / total_b for k in vocab])

    if method == "kld":
        return float(np.sum(p * np.log(p / q)))
    if method == "jsd":
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        return 0.5 * kl_pm + 0.5 * kl_qm
    raise ValueError(f"unknown method: {method}")


def _get_bpe_merges(corpus: Iterable[str], vocab_size: int, **trainer_kwargs) -> list[tuple[str, str]]:
    """Train a BPE tokenizer on `corpus` and return its ordered merge list."""
    tokenizer = BPETokenizer(corpus, vocab_size=vocab_size, **trainer_kwargs).tokenizer
    merges = json.loads(tokenizer.to_str())["model"]["merges"]
    # return [tuple(m.split(" ", 1)) for m in merges]
    return merges


def bpe_merge_rank_correlation(
    corpus_a: Iterable[bytes],
    corpus_b: Iterable[bytes],
    vocab_size: int = 10000,
    method: Literal["spearman", "kendall"] = "kendall",
    min_shared: int = 50,
) -> float:
    """Rank correlation of shared BPE merge rules between two corpora.

    Trains a BPE tokenizer on each corpus independently and computes the rank
    correlation of merge rules that appear in both merge tables. Returns a value
    in [-1, 1]; higher means more similar domain.

    Kendall's τ (default) counts concordant vs discordant merge-order pairs and
    is more robust to the heavy-tailed rank distribution of BPE tables. Spearman
    is sensitive to the magnitude of rank disagreements.

    Requires: pip install tokenizers
    """
    from scipy.stats import kendalltau, spearmanr

    merges_a = _get_bpe_merges(
        (doc.decode("utf-8", errors="replace") for doc in corpus_a), vocab_size
    )
    merges_b = _get_bpe_merges(
        (doc.decode("utf-8", errors="replace") for doc in corpus_b), vocab_size
    )

    rank_a = {m: i for i, m in enumerate(merges_a)}
    rank_b = {m: i for i, m in enumerate(merges_b)}

    shared = set(rank_a) & set(rank_b)
    if len(shared) < min_shared:
        raise ValueError(
            f"only {len(shared)} shared merge rules (< min_shared={min_shared}); "
            "increase vocab_size or corpus size"
        )

    ra = [rank_a[m] for m in shared]
    rb = [rank_b[m] for m in shared]

    if method == "kendall":
        stat, _ = kendalltau(ra, rb)
    elif method == "spearman":
        stat, _ = spearmanr(ra, rb)
    else:
        raise ValueError(f"unknown method: {method}")
    return float(stat)


def bpe_overlap(
    corpus_a: Iterable[bytes],
    corpus_b: Iterable[bytes],
    *,
    vocab_size: int = 1_000_000,
    min_freq_fract: float = 1e-7,
    k_steps: int = 20,
    return_dict=False, 
    **kwargs,
) -> dict:
    """Asymmetric BPE-coverage metric: how well corpus_a's merge rules compress corpus_b.

    1. Train a long BPE merge list on corpus_a, stopping when merge frequency falls
       below min_freq_fract * total_bytes_a.
    2. At k_steps log-spaced values of k, build a tokenizer with the top-k merges,
       tokenize corpus_b, and record bytes-per-token (BPT).
    3. Train BPE on corpus_b itself and measure the skyline BPT on corpus_b.
    4. Normalize: BPT_rel(k) = (BPT(k) - 1) / (BPT_skyline - 1), so 0 means no
       compression gain beyond single-byte tokens and 1 means as efficient as training
       on b itself.
    5. Compute AUC of BPT_rel over k/max_k in [0, 1].

    Returns a dict with:
        auc         — area under the BPT_rel curve (higher = more domain overlap)
        bpt_skyline — BPT of a tokenizer trained and evaluated on corpus_b
        curve       — list of (k, bpt_rel) pairs
        n_merges    — number of merges learned from corpus_a
    """
    docs_a = list(corpus_a)
    docs_b = list(corpus_b)
    if not docs_a or not docs_b:
        raise ValueError("both corpora must be non-empty")

    str_a = [doc.decode("utf-8", errors="replace") for doc in docs_a]

    size_a = sum(len(doc) for doc in docs_a)
    min_freq = max(1, int(size_a * min_freq_fract))
    merges_a = _get_bpe_merges(str_a, vocab_size=vocab_size, min_frequency=min_freq, **kwargs)
    max_k = len(merges_a)
    if max_k == 0:
        raise ValueError("no BPE merges learned from corpus_a; try a smaller min_freq_fract")

    # Log-spaced k schedule from 1 to max_k
    k_values = list(map(int, np.unique(np.geomspace(1, max_k, k_steps).round().astype(int))))
    if k_values[-1] != max_k:
        k_values.append(max_k)

    # Sweep k incrementally: extend() only processes the delta on the Python side
    tok = BPEFromMerges()
    prev_k = 0
    curve_k, curve_bpt = [], []
    for k in k_values:
        tok.extend(merges_a[prev_k:k])
        curve_k.append(k)
        curve_bpt.append(tok.measure_bpt(docs_b))
        prev_k = k

    # Skyline: BPE trained and evaluated on corpus_b
    size_b = sum(len(doc) for doc in docs_b)
    min_freq_b = max(1, int(size_b * min_freq_fract))
    str_b = [doc.decode("utf-8", errors="replace") for doc in docs_b]
    merges_b = _get_bpe_merges(str_b, vocab_size=vocab_size, min_frequency=min_freq_b, **kwargs)
    bpt_skyline = BPEFromMerges(merges_b).measure_bpt(docs_b)

    denom = bpt_skyline - 1.0
    if denom <= 0:
        raise ValueError(f"bpt_skyline={bpt_skyline:.3f} <= 1.0; corpus_b may be too small to train BPE")

    bpt_rel = [(bpt - 1.0) / denom for bpt in curve_bpt]
    k_norm = [k / max_k for k in curve_k]
    auc = float(np.trapezoid(bpt_rel, k_norm))

    if return_dict: 
        return {
            "auc": auc,
            "bpt_skyline": bpt_skyline,
            "curve": list(zip(curve_k, bpt_rel)),
            "n_merges": max_k,
        }
    return auc

def normalized_compression_distance(
    corpus_a: Iterable[bytes],
    corpus_b: Iterable[bytes],
    compressor: Callable[[bytes], bytes] | None = None,
    *,
    symmetric: bool = False,
) -> float:
    """NCD(a, b) = (C(ab) - min(C(a), C(b))) / max(C(a), C(b))

    Documents in each corpus are concatenated before compression.

    Uses zlib by default; pass a different `compressor` (e.g. bz2.compress,
    lzma.compress) to compare under a different compression scheme.

    symmetric=True averages C(ab) and C(ba) before computing the ratio,
    removing the order-dependence of concatenation.  This adds one extra
    compression call but produces a true metric.

    Note: result is not strictly bounded to [0, 1]; for very short inputs the
    compressor's fixed overhead can push it above 1.0.
    """
    a = b"".join(corpus_a)
    b_ = b"".join(corpus_b)

    if len(a) < _NCD_SHORT_TEXT_THRESHOLD or len(b_) < _NCD_SHORT_TEXT_THRESHOLD:
        _log.warning(
            "NCD inputs are short (%d, %d bytes); compressor overhead may push "
            "the result above 1.0 and reduce reliability.",
            len(a),
            len(b_),
        )

    compress = compressor or zlib.compress

    c_a = len(compress(a))
    c_b = len(compress(b_))
    if symmetric:
        c_ab = (len(compress(a + b_)) + len(compress(b_ + a))) / 2
    else:
        c_ab = len(compress(a + b_))

    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)


def normalized_compression_distance_asymmetric(
    corpus_a: Iterable[bytes],
    corpus_b: Iterable[bytes],
    compressor: Callable[[bytes], bytes] | None = None,
) -> float:
    """Asymmetric NCD suited for the case where |a| >> |b|.

    Documents in each corpus are concatenated before compression.

    Returns (C(ab) - C(a)) / C(b): the fraction of b's information that is
    not already captured in a, normalised by b's own complexity.  Useful when
    a is a large reference corpus and b is a small query set; in that regime
    the standard symmetric NCD is dominated by a's size.

    Interpretation: values near 0 mean b is well-predicted by a; values near
    (or above) 1 mean b contains information largely absent from a.
    """
    a = b"".join(corpus_a)
    b_ = b"".join(corpus_b)

    if len(b_) < _NCD_SHORT_TEXT_THRESHOLD:
        _log.warning(
            "Asymmetric NCD: corpus_b is short (%d bytes); C(b) is dominated by "
            "compressor overhead and the result may be unreliable.",
            len(b_),
        )

    compress = compressor or zlib.compress
    c_a = len(compress(a))
    c_b = len(compress(b_))
    c_ab = len(compress(a + b_))
    return (c_ab - c_a) / c_b

# ---- wrapper for sampled variants of the distance metrics --------------------

def _sample_to_size(docs: list[bytes], target: int, rng: random.Random) -> list[bytes]:
    """Sample documents (without replacement) until cumulative byte length >= target."""
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    sample: list[bytes] = []
    total = 0
    for i in indices:
        sample.append(docs[i])
        total += len(docs[i])
        if total >= target:
            break
    return sample


def sampled_distance(
    corpus_a: Iterable[bytes],
    corpus_b: Iterable[bytes],
    metric: Callable[[list[bytes], list[bytes]], float],
    *,
    sample_size_per_iteration: int = 100_000,
    k: int = 30,
    threshold: float = 0.01,
    max_iterations: int = 1000,
    return_window: bool = False,
    seed: int | None = None,
    show_progress: bool = False,
) -> dict:
    """Estimate a corpus-level metric via convergent random sampling.

    Each iteration draws a random subset of documents from each corpus whose
    total byte size is at least `sample_size_per_iteration`, then evaluates
    `metric` on the two subsets.  Stops when the rolling standard deviation of
    the last `k` cumulative-mean estimates drops below `threshold`, or after
    `max_iterations`.

    Convergence is tested on running cumulative means rather than raw sample
    values: std(cumulative_means[-k:]) measures how much the overall estimate
    is still shifting as evidence accumulates, not how noisy individual samples
    are.

    Args:
        corpus_a, corpus_b:        iterables of byte strings (materialised internally).
        metric:                    callable (sample_a, sample_b) -> float, where each
                                   sample is a list[bytes].
        sample_size_per_iteration: target cumulative byte size of each per-corpus sample.
        k:                         window size for convergence testing (last k cumulative means).
        threshold:                 stop when std(last k cumulative means) < threshold.
        max_iterations:            hard upper bound on iterations.
        return_window:             if True, include all iteration values in the result.
        seed:                      RNG seed for reproducibility.

    Returns a dict with:
        mean          — mean metric value over all iterations
        std           — std of the last k cumulative-mean estimates (the convergence window)
        n_iterations  — total number of iterations run
        converged     — True if the std threshold was reached before max_iterations
        values        — all iteration values (only present when return_window=True)
    """
    rng = random.Random(seed)
    docs_a = list(corpus_a)
    docs_b = list(corpus_b)
    if not docs_a or not docs_b:
        raise ValueError("both corpora must be non-empty")

    values: list[float] = []
    cumulative_means: list[float] = []
    converged = False

    if show_progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=max_iterations, desc="sampled_distance", leave=False)
    else:
        pbar = None

    for _ in range(max_iterations):
        sample_a = _sample_to_size(docs_a, sample_size_per_iteration, rng)
        sample_b = _sample_to_size(docs_b, sample_size_per_iteration, rng)
        values.append(metric(sample_a, sample_b))
        cumulative_means.append(float(np.mean(values)))
        if len(cumulative_means) >= k:
            window_std = float(np.std(cumulative_means[-k:], ddof=1))
            if pbar is not None:
                pbar.set_postfix(std=f"{window_std:.4f}")
            if window_std < threshold:
                converged = True
                if pbar is not None:
                    pbar.update(1)
                break
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    cm_window = cumulative_means[-k:] if len(cumulative_means) >= k else cumulative_means
    result = {
        "mean": cumulative_means[-1],
        "std": float(np.std(cm_window, ddof=1)) if len(cm_window) > 1 else float("nan"),
        "n_iterations": len(values),
        "converged": converged,
    }
    if return_window:
        result["values"] = list(values)
    return result

