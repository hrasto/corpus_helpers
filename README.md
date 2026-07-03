# corpus_helpers

This repo contains various functions that should be helpful for work on text corpora. 

**Overview**

The packages contains the following modules: 

`readers`
- functions to load/stream/preprocess normalize text files

`visualize`
- visualize documents in t-sne, optionally with 'group' annotations
- visualize/estimate Zipf's law

`domain_distance`
- text similarity metrics based on various (surface) features

`partitioning`
- to partition a text corpus into regions which contain similar documents
- degradation testing

`tokenizers`
- wrappers around huggingface tokenizers to provide fast leftmost-longest segmentation
- potentially could also include graphical segmentation

`vocabulary`
- functions for vocabulary construction (bpe, unigram, pickybpe, sentencepiece, wordpiece, morfessor, etc.)
- group-frequency filtering

**Installing dependencies/Running tests**

```bash
pip install -r requirements.txt
pip install pytest
pytest tests/
```

---

**Table of contents**

- [Readers](#readers)
- [Vocabulary](#vocabulary)
- [Visualize](#visualize)
- [Domain distance](#domain-distance)
- [Partitioning](#partitioning)
- [Tokenizers](#tokenizers)

---

## Readers

`GroupView` organizes files into a lazy iterable of restartable readers, suitable for passing directly to `VocabFilter`.

```python
from corpus_helpers.read import GroupView, split_line, lower

# doc mode: one reader per file
docs = GroupView(["a.txt", "b.txt"], preprocessors=[split_line])

# group mode: one reader per group, files within a group are chained
docs = GroupView([["a.txt", "b.txt"], ["c.txt"]], preprocessors=[split_line, lower])
```

`restartable_file_reader` and `chain_preprocessors` are the lower-level building blocks used internally by `GroupView`.

---

## Vocabulary

`VocabFilter` accumulates term and document/group frequencies across a corpus and supports threshold-based filtering.

```python
from corpus_helpers.read import GroupView
from corpus_helpers.vocab import VocabFilter, build_bpe_vocab

# build a vocab counter from one document (iterable of strings)
vocab_builder = lambda doc: build_bpe_vocab(doc)

# flat: df counts per document
vf = VocabFilter(GroupView(["a.txt", "b.txt"]), vocab_builder)

# nested: df counts per group
vf = VocabFilter(GroupView([["a.txt", "b.txt"], ["c.txt"]]), vocab_builder)

vocab, cutoff = vf.get_vocab(max_size=50000, min_df=2, order_by="df")
```

Available vocab builders: `build_bpe_vocab`, `build_picky_bpe_vocab`, `build_unigram_vocab`, `build_morfessor_vocab`.

---

## Visualize

*TODO*

---

## Domain distance

`metrics.py` provides surface-level distance metrics between two text corpora.
Each corpus is an iterable of strings (documents or lines).

### Lexical metrics

```python
from corpus_helpers.metrics import (
    lexical_overlap,
    lexical_divergence,
    ngram_counts,
    ngram_overlap,
    ngram_divergence,
)

a = ["the cat sat on the mat", "a quick brown fox"]
b = ["the dog lay on the rug", "a lazy brown dog"]

# Jaccard overlap of word vocabularies (0 = disjoint, 1 = identical)
lexical_overlap(a, b)           # → float in [0, 1]

# JSD between word-frequency distributions (0 = identical)
lexical_divergence(a, b)                     # JSD (default)
lexical_divergence(a, b, method="kl")        # KL divergence (asymmetric)

# Character trigram overlap / divergence
c3 = ngram_counts(a, n=3, unit="char")
d3 = ngram_counts(b, n=3, unit="char")
ngram_overlap(c3, d3)
ngram_divergence(c3, d3, method="jsd")
```

### BPE merge-rank correlation

```python
from corpus_helpers.metrics import bpe_merge_rank_correlation

score = bpe_merge_rank_correlation(a, b, vocab_size=8000, method="kendall")
# → float in [-1, 1]; higher means more similar BPE structure
```

### Normalized Compression Distance

```python
from corpus_helpers.metrics import normalized_compression_distance

ncd = normalized_compression_distance("hello world", "hello there")
# → float near 0 for similar texts, near 1 for unrelated ones
# Pass a custom compressor (e.g. bz2.compress) as the third argument.
```

---

## Partitioning

*TODO*

---

## Tokenizers

*TODO*
