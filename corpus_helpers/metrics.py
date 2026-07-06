"""Domain-distance metrics between two text corpora.

Each corpus is represented as an iterable of byte strings (documents/lines).
Metrics operate on frequency distributions derived from those documents, either
at the lexical (whitespace-token) level or the n-gram (character or word) level.

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
from .tokenizers2 import BPETokenizer
import numpy as np

Unit = Literal["char", "word"]

_log = logging.getLogger(__name__)

# Texts shorter than this (in bytes) may produce unreliable NCD values
# because the compressor's fixed-overhead bytes dominate the compressed size.
_NCD_SHORT_TEXT_THRESHOLD = 50

# letters | digit runs | punctuation/symbols — never mixes categories across boundaries
_WORD_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+", re.UNICODE)


def _word_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def _ngram_counts(corpus: Iterable[bytes], n: int = 1, unit: Unit = "word") -> Counter:
    """Count n-grams over a corpus of documents.

    unit="char" splits each document into characters before forming n-grams;
    unit="word" splits on whitespace.
    """
    counts: Counter = Counter()
    for doc in corpus:
        text = doc.decode("utf-8", errors="replace")
        tokens = list(text) if unit == "char" else _word_tokenize(text)
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
    method: Literal["jsd", "kl"] = "jsd",
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

    if method == "kl":
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
    return [tuple(m.split(" ", 1)) for m in merges]


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

    merges_a = _train_bpe(
        (doc.decode("utf-8", errors="replace") for doc in corpus_a), vocab_size
    )
    merges_b = _train_bpe(
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


def normalized_compression_distance(
    doc_a: bytes,
    doc_b: bytes,
    compressor: Callable[[bytes], bytes] | None = None,
    *,
    symmetric: bool = False,
) -> float:
    """NCD(a, b) = (C(ab) - min(C(a), C(b))) / max(C(a), C(b))

    Uses zlib by default; pass a different `compressor` (e.g. bz2.compress,
    lzma.compress) to compare under a different compression scheme.

    symmetric=True averages C(ab) and C(ba) before computing the ratio,
    removing the order-dependence of concatenation.  This adds one extra
    compression call but produces a true metric.

    Note: result is not strictly bounded to [0, 1]; for very short texts the
    compressor's fixed overhead can push it above 1.0.
    """
    if len(doc_a) < _NCD_SHORT_TEXT_THRESHOLD or len(doc_b) < _NCD_SHORT_TEXT_THRESHOLD:
        _log.warning(
            "NCD inputs are short (%d, %d bytes); compressor overhead may push "
            "the result above 1.0 and reduce reliability.",
            len(doc_a),
            len(doc_b),
        )

    compress = compressor or zlib.compress

    c_a = len(compress(doc_a))
    c_b = len(compress(doc_b))
    if symmetric:
        c_ab = (len(compress(doc_a + doc_b)) + len(compress(doc_b + doc_a))) / 2
    else:
        c_ab = len(compress(doc_a + doc_b))

    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)


def normalized_compression_distance_asymmetric(
    doc_a: bytes,
    doc_b: bytes,
    compressor: Callable[[bytes], bytes] | None = None,
) -> float:
    """Asymmetric NCD suited for the case where |a| >> |b|.

    Returns (C(ab) - C(a)) / C(b): the fraction of b's information that is
    not already captured in a, normalised by b's own complexity.  Useful when
    a is a large reference text and b is a short query document; in that
    regime the standard symmetric NCD is dominated by a's size.

    Interpretation: values near 0 mean b is well-predicted by a; values near
    (or above) 1 mean b contains information largely absent from a.
    """
    if len(doc_b) < _NCD_SHORT_TEXT_THRESHOLD:
        _log.warning(
            "Asymmetric NCD: doc_b is short (%d bytes); C(b) is dominated by "
            "compressor overhead and the result may be unreliable.",
            len(doc_b),
        )

    compress = compressor or zlib.compress
    c_a = len(compress(doc_a))
    c_b = len(compress(doc_b))
    c_ab = len(compress(doc_a + doc_b))
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
) -> dict:
    """Estimate a corpus-level metric via convergent random sampling.

    Each iteration draws a random subset of documents from each corpus whose
    total byte size is at least `sample_size_per_iteration`, then evaluates
    `metric` on the two subsets.  Stops when the rolling standard deviation of
    the last `k` values drops below `threshold`, or after `max_iterations`.

    Args:
        corpus_a, corpus_b:        iterables of byte strings (materialised internally).
        metric:                    callable (sample_a, sample_b) -> float, where each
                                   sample is a list[bytes].
        sample_size_per_iteration: target cumulative byte size of each per-corpus sample.
        k:                         rolling window size for convergence testing.
        threshold:                 stop when std(last k values) < threshold.
        max_iterations:            hard upper bound on iterations.
        return_window:             if True, include all iteration values in the result.
        seed:                      RNG seed for reproducibility.

    Returns a dict with:
        mean          — mean metric value over all iterations
        std           — std of the last k values (the convergence window)
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
    converged = False

    for _ in range(max_iterations):
        sample_a = _sample_to_size(docs_a, sample_size_per_iteration, rng)
        sample_b = _sample_to_size(docs_b, sample_size_per_iteration, rng)
        values.append(metric(sample_a, sample_b))
        if len(values) >= k:
            window_std = float(np.std(values[-k:], ddof=1))
            if window_std < threshold:
                converged = True
                break

    window = values[-k:] if len(values) >= k else values
    result = {
        "mean": float(np.mean(values)),
        "std": float(np.std(window, ddof=1)) if len(window) > 1 else float("nan"),
        "n_iterations": len(values),
        "converged": converged,
    }
    if return_window:
        result["values"] = list(values)
    return result
