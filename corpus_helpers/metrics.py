"""Domain-distance metrics between two text corpora.

Each corpus is represented as an iterable of strings (documents/lines). Metrics
operate on frequency distributions derived from those documents, either at
the lexical (whitespace-token) level or the n-gram (character or word) level.

BPE metrics additionally require the `tokenizers` package (HuggingFace).
"""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from typing import Callable, Iterable, Literal

import numpy as np

Unit = Literal["char", "word"]

# letters | digit runs | punctuation/symbols — never mixes categories across boundaries
_WORD_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+", re.UNICODE)


def _word_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def ngram_counts(corpus: Iterable[str], n: int = 1, unit: Unit = "word") -> Counter:
    """Count n-grams over a corpus of documents.

    unit="char" splits each document into characters before forming n-grams;
    unit="word" splits on whitespace.
    """
    counts: Counter = Counter()
    for doc in corpus:
        tokens = list(doc) if unit == "char" else _word_tokenize(doc)
        counts.update(_ngrams(tokens, n))
    return counts


def _to_prob(counts: Counter) -> dict:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("empty corpus: cannot form a probability distribution")
    return {k: v / total for k, v in counts.items()}


def ngram_overlap(counts_a: Counter, counts_b: Counter) -> float:
    """Jaccard overlap between the n-gram vocabularies of two corpora."""
    keys_a, keys_b = set(counts_a), set(counts_b)
    union = keys_a | keys_b
    if not union:
        return 0.0
    return len(keys_a & keys_b) / len(union)


def ngram_divergence(
    counts_a: Counter, counts_b: Counter, method: Literal["jsd", "kl"] = "jsd"
) -> float:
    """KL or Jensen-Shannon divergence between two n-gram frequency distributions.

    KL divergence is computed over the support of counts_a, smoothed against
    counts_b's vocabulary (add-one) to avoid undefined log(0/x) terms.
    """
    vocab = set(counts_a) | set(counts_b)
    total_a = sum(counts_a.values()) + len(vocab)
    total_b = sum(counts_b.values()) + len(vocab)
    p = np.array([(counts_a.get(k, 0) + 1) / total_a for k in vocab])
    q = np.array([(counts_b.get(k, 0) + 1) / total_b for k in vocab])

    if method == "kl":
        return float(np.sum(p * np.log(p / q)))
    if method == "jsd":
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        return 0.5 * kl_pm + 0.5 * kl_qm
    raise ValueError(f"unknown method: {method}")


def lexical_overlap(corpus_a: Iterable[str], corpus_b: Iterable[str]) -> float:
    """Jaccard overlap between the word-level vocabularies of two corpora."""
    vocab_a = {w for doc in corpus_a for w in _word_tokenize(doc)}
    vocab_b = {w for doc in corpus_b for w in _word_tokenize(doc)}
    union = vocab_a | vocab_b
    if not union:
        return 0.0
    return len(vocab_a & vocab_b) / len(union)


def lexical_divergence(
    corpus_a: Iterable[str], corpus_b: Iterable[str], method: Literal["jsd", "kl"] = "jsd"
) -> float:
    """KL or JSD divergence between word-frequency distributions of two corpora."""
    counts_a = ngram_counts(corpus_a, n=1, unit="word")
    counts_b = ngram_counts(corpus_b, n=1, unit="word")
    return ngram_divergence(counts_a, counts_b, method=method)


def _train_bpe(corpus: Iterable[str], vocab_size: int) -> list[tuple[str, str]]:
    """Train a BPE tokenizer on `corpus` and return its ordered merge list."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer
    except ImportError as e:
        raise ImportError("`tokenizers` is required (pip install tokenizers)") from e

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=False)
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    merges = json.loads(tokenizer.to_str())["model"]["merges"]
    return [tuple(m.split(" ", 1)) for m in merges]


def bpe_merge_rank_correlation(
    corpus_a: Iterable[str],
    corpus_b: Iterable[str],
    vocab_size: int = 8000,
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

    merges_a = _train_bpe(corpus_a, vocab_size)
    merges_b = _train_bpe(corpus_b, vocab_size)

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
    text_a: str, text_b: str, compressor: Callable[[bytes], bytes] | None = None
) -> float:
    """NCD(a, b) = (C(ab) - min(C(a), C(b))) / max(C(a), C(b))

    Uses zlib by default; pass a different `compressor` (e.g. bz2.compress,
    lzma.compress) to compare under a different compression scheme.
    """
    compress = compressor or zlib.compress
    a_bytes, b_bytes = text_a.encode("utf-8"), text_b.encode("utf-8")

    c_a = len(compress(a_bytes))
    c_b = len(compress(b_bytes))
    c_ab = len(compress(a_bytes + b_bytes))

    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)
