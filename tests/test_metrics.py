"""Unit tests for corpus_helpers.metrics."""

import logging
import zlib
from collections import Counter

import pytest

from corpus_helpers.metrics import (
    corpus_ncd,
    lexical_divergence,
    lexical_overlap,
    ngram_counts,
    ngram_divergence,
    ngram_overlap,
    normalized_compression_distance,
    normalized_compression_distance_asymmetric,
)


# ---------------------------------------------------------------------------
# ngram_counts
# ---------------------------------------------------------------------------


def test_ngram_counts_word_unigrams():
    counts = ngram_counts(["hello world", "hello"], n=1, unit="word")
    assert counts[("hello",)] == 2
    assert counts[("world",)] == 1


def test_ngram_counts_char_bigrams():
    counts = ngram_counts(["ab"], n=2, unit="char")
    assert counts[("a", "b")] == 1
    assert len(counts) == 1


def test_ngram_counts_bigrams_word():
    counts = ngram_counts(["a b c"], n=2, unit="word")
    assert counts[("a", "b")] == 1
    assert counts[("b", "c")] == 1


def test_ngram_counts_empty_corpus():
    counts = ngram_counts([], n=1, unit="word")
    assert len(counts) == 0


def test_ngram_counts_empty_document():
    counts = ngram_counts([""], n=1, unit="word")
    assert len(counts) == 0


# ---------------------------------------------------------------------------
# ngram_overlap
# ---------------------------------------------------------------------------


def test_ngram_overlap_identical():
    c = Counter({("a",): 3, ("b",): 1})
    assert ngram_overlap(c, c) == pytest.approx(1.0)


def test_ngram_overlap_disjoint():
    ca = Counter({("a",): 1})
    cb = Counter({("b",): 1})
    assert ngram_overlap(ca, cb) == pytest.approx(0.0)


def test_ngram_overlap_partial():
    ca = Counter({("a",): 1, ("b",): 1})
    cb = Counter({("b",): 1, ("c",): 1})
    # shared = {b}, union = {a, b, c} → 1/3
    assert ngram_overlap(ca, cb) == pytest.approx(1 / 3)


def test_ngram_overlap_both_empty():
    assert ngram_overlap(Counter(), Counter()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ngram_divergence
# ---------------------------------------------------------------------------


def test_ngram_divergence_identical_jsd():
    c = Counter({("a",): 10, ("b",): 5})
    assert ngram_divergence(c, c, method="jsd") == pytest.approx(0.0, abs=1e-10)


def test_ngram_divergence_jsd_symmetric():
    ca = Counter({("a",): 10, ("b",): 2})
    cb = Counter({("b",): 8, ("c",): 1})
    assert ngram_divergence(ca, cb, "jsd") == pytest.approx(
        ngram_divergence(cb, ca, "jsd"), rel=1e-9
    )


def test_ngram_divergence_kl_positive():
    ca = Counter({("a",): 10, ("b",): 2})
    cb = Counter({("b",): 8, ("c",): 1})
    assert ngram_divergence(ca, cb, "kl") > 0


def test_ngram_divergence_kl_asymmetric():
    ca = Counter({("a",): 10, ("b",): 2})
    cb = Counter({("b",): 8, ("c",): 1})
    # KL is not generally symmetric
    assert ngram_divergence(ca, cb, "kl") != pytest.approx(
        ngram_divergence(cb, ca, "kl"), rel=1e-6
    )


def test_ngram_divergence_unknown_method():
    c = Counter({("a",): 1})
    with pytest.raises(ValueError, match="unknown method"):
        ngram_divergence(c, c, method="cosine")  # type: ignore[arg-type]


def test_ngram_divergence_empty_both_raises():
    with pytest.raises(ValueError, match="empty"):
        ngram_divergence(Counter(), Counter())


def test_ngram_divergence_smoothing_changes_result():
    ca = Counter({("a",): 10, ("b",): 2})
    cb = Counter({("b",): 8, ("c",): 1})
    d1 = ngram_divergence(ca, cb, smoothing=1.0)
    d2 = ngram_divergence(ca, cb, smoothing=0.01)
    assert d1 != pytest.approx(d2, rel=1e-6)


def test_ngram_divergence_smoothing_zero_limit_positive():
    # Even with very small smoothing the JSD should remain non-negative
    ca = Counter({("a",): 10})
    cb = Counter({("b",): 10})
    assert ngram_divergence(ca, cb, smoothing=1e-9) >= 0


# ---------------------------------------------------------------------------
# lexical_overlap
# ---------------------------------------------------------------------------


def test_lexical_overlap_identical():
    corpus = ["the cat sat", "a quick fox"]
    assert lexical_overlap(corpus, corpus) == pytest.approx(1.0)


def test_lexical_overlap_disjoint():
    a = ["alpha beta"]
    b = ["gamma delta"]
    assert lexical_overlap(a, b) == pytest.approx(0.0)


def test_lexical_overlap_empty():
    assert lexical_overlap([], []) == pytest.approx(0.0)


def test_lexical_overlap_partial():
    a = ["cat dog bird"]
    b = ["cat fish snake"]
    # shared = {cat}, union = {cat, dog, bird, fish, snake} → 1/5
    assert lexical_overlap(a, b) == pytest.approx(1 / 5)


# ---------------------------------------------------------------------------
# lexical_divergence
# ---------------------------------------------------------------------------


def test_lexical_divergence_identical_near_zero():
    corpus = ["the cat sat on the mat"] * 10
    assert lexical_divergence(corpus, corpus) == pytest.approx(0.0, abs=1e-10)


def test_lexical_divergence_different_positive():
    a = ["alpha beta gamma"] * 5
    b = ["delta epsilon zeta"] * 5
    assert lexical_divergence(a, b) > 0


# ---------------------------------------------------------------------------
# normalized_compression_distance
# ---------------------------------------------------------------------------


def test_ncd_identical_near_zero():
    # zlib won't reach 0 even for identical inputs; 0.2 is a realistic ceiling
    text = "the quick brown fox jumps over the lazy dog " * 20
    ncd = normalized_compression_distance(text, text)
    assert ncd < 0.2


def test_ncd_unrelated_higher():
    a = "aaaa bbbb cccc dddd eeee " * 50
    b = "1111 2222 3333 4444 5555 " * 50
    ncd_same = normalized_compression_distance(a, a)
    ncd_diff = normalized_compression_distance(a, b)
    assert ncd_diff > ncd_same


def test_ncd_custom_compressor():
    import bz2

    text = "hello world " * 100
    ncd = normalized_compression_distance(text, text, compressor=bz2.compress)
    assert ncd == pytest.approx(0.0, abs=0.1)


def test_ncd_returns_float():
    result = normalized_compression_distance("abc", "xyz")
    assert isinstance(result, float)


def test_ncd_symmetric_flag_produces_float():
    a = "the cat sat on the mat " * 30
    b = "the dog lay on the rug " * 30
    result = normalized_compression_distance(a, b, symmetric=True)
    assert isinstance(result, float)


def test_ncd_symmetric_vs_asymmetric_order():
    # For non-symmetric mode, swapping inputs may give a different result
    # (concatenation order matters).  Symmetric mode should be invariant.
    a = "alpha beta gamma " * 40
    b = "delta epsilon zeta " * 40
    sym_ab = normalized_compression_distance(a, b, symmetric=True)
    sym_ba = normalized_compression_distance(b, a, symmetric=True)
    assert sym_ab == pytest.approx(sym_ba, rel=1e-9)


def test_ncd_short_text_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="corpus_helpers.metrics"):
        normalized_compression_distance("hi", "there")
    assert any("short" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# normalized_compression_distance_asymmetric
# ---------------------------------------------------------------------------


def test_ncd_asym_returns_float():
    a = "reference text " * 100
    b = "query sentence " * 5
    result = normalized_compression_distance_asymmetric(a, b)
    assert isinstance(result, float)


def test_ncd_asym_familiar_b_near_zero():
    # b is essentially a substring of a, so (C(ab) - C(a)) should be small
    base = "the quick brown fox jumps over the lazy dog " * 50
    query = "the quick brown fox " * 3
    score = normalized_compression_distance_asymmetric(base, query)
    assert score < 0.5


def test_ncd_asym_novel_b_higher():
    base = "alpha beta gamma delta " * 50
    familiar = "alpha beta " * 5
    novel = "zeta eta theta iota " * 5
    assert normalized_compression_distance_asymmetric(base, novel) > \
           normalized_compression_distance_asymmetric(base, familiar)


def test_ncd_asym_short_b_warning(caplog):
    a = "long reference text " * 100
    with caplog.at_level(logging.WARNING, logger="corpus_helpers.metrics"):
        normalized_compression_distance_asymmetric(a, "short")
    assert any("short" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# corpus_ncd
# ---------------------------------------------------------------------------

_CORPUS_A = ["the cat sat on the mat " * 10] * 20
_CORPUS_B = ["the dog lay on the rug " * 10] * 20


def test_corpus_ncd_returns_expected_keys():
    result = corpus_ncd(_CORPUS_A, _CORPUS_B, k=5, max_samples=10, seed=0)
    assert set(result) == {"mean", "std", "n_samples", "converged"}


def test_corpus_ncd_mean_in_range():
    result = corpus_ncd(_CORPUS_A, _CORPUS_B, k=5, max_samples=20, seed=42)
    assert 0.0 <= result["mean"] <= 2.0


def test_corpus_ncd_n_samples_bounded():
    result = corpus_ncd(_CORPUS_A, _CORPUS_B, k=5, max_samples=15, seed=1)
    assert result["n_samples"] <= 15


def test_corpus_ncd_convergence_flag():
    # Identical corpora should converge quickly (std ≈ 0)
    result = corpus_ncd(_CORPUS_A, _CORPUS_A, k=5, threshold=0.1, max_samples=200, seed=7)
    assert result["converged"] is True


def test_corpus_ncd_no_convergence_hits_max():
    # k > max_samples means the convergence window never fills, so all
    # max_samples pairs are drawn and converged is False.
    result = corpus_ncd(_CORPUS_A, _CORPUS_B, k=20, threshold=0.0, max_samples=8, seed=3)
    assert result["n_samples"] == 8
    assert result["converged"] is False


def test_corpus_ncd_reproducible_with_seed():
    r1 = corpus_ncd(_CORPUS_A, _CORPUS_B, k=5, max_samples=20, seed=99)
    r2 = corpus_ncd(_CORPUS_A, _CORPUS_B, k=5, max_samples=20, seed=99)
    assert r1["mean"] == r2["mean"]
    assert r1["n_samples"] == r2["n_samples"]


def test_corpus_ncd_empty_corpus_raises():
    with pytest.raises(ValueError, match="non-empty"):
        corpus_ncd([], _CORPUS_B)


def test_corpus_ncd_accepts_generators():
    gen_a = (doc for doc in _CORPUS_A)
    gen_b = (doc for doc in _CORPUS_B)
    result = corpus_ncd(gen_a, gen_b, k=5, max_samples=10, seed=0)
    assert "mean" in result
