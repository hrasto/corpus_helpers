# corpus_helpers

This repo contains various functions that should be helpful for work on text corpora. 

**Overview**

The packages contains the following modules: 

`readers`
- functions to load/stream/preprocess normalize text files

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

`partition.py` provides a pipeline for splitting a text corpus into topically coherent regions. A full walkthrough is in [`notebooks/partition_demo.ipynb`](notebooks/partition_demo.ipynb).

### Typical pipeline

**1. Vectorise and fit a topic model**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from corpus_helpers import partition

vectorizer = CountVectorizer(min_df=10, max_df=0.8, max_features=10000)
docs_vect = vectorizer.fit_transform(docs)

model = partition.fit_topic_model(docs_vect, LatentDirichletAllocation, n_components=14, random_state=0)
```

**2. Cluster documents in topic space**

```python
docs_latent = model.transform(docs_vect)   # (n_docs, n_topics)
assign = partition.partition(docs_latent, n_clusters=14, seed=0)
```

**3. Compute region sizes and iteratively split the largest region**

```python
file_sizes = [len(d.encode()) for d in docs]   # bytes per document

region_sizes = partition.get_region_sizes(assign, file_sizes)
# → {region_id: size_in_MB, …}

assign2 = partition.split_largest_region(assign, docs_latent, region_sizes, seed=0)
# largest region is re-clustered into two; one half keeps the original id,
# the other gets max(assign)+1
```

**4. Find the most / least similar subsets of regions**

```python
lo, hi = partition.make_subsets(assign2, docs_latent, subset_size=3)
# lo: tuple of region ids with lowest mean pairwise cosine distance (most similar)
# hi: tuple of region ids with highest mean pairwise cosine distance (most dissimilar)
```

### Saving and reloading a topic model

```python
partition.save_topic_model(model, vectorizer, path="./my_model")
model, vectorizer = partition.load_topic_model("./my_model")
```

---

## Tokenizers

`tokenizers2.py` provides tokenizer wrappers built on top of [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers), all operating on byte-level representations. The module is named `tokenizers2` to avoid shadowing the HF library.

All classes share a common interface via `BaseTokenizer`: construct with a text iterable to train immediately, then call `.tokenize(text)` or `.encode(text)` to get a list of token strings.

### Leftmost-longest (maximum matching)

```python
from corpus_helpers.tokenizers2 import LeftmostLongestTokenizer

tok = LeftmostLongestTokenizer(terms=my_vocab)   # vocab: list/dict of strings
tok.tokenize("hello world")   # → list of token strings
```

Greedily picks the longest matching token at each position. Relies on HF's BPE fallback (no merge list), so no training corpus is needed — just supply a vocabulary.

### Wrapping an external tokenizer

```python
from corpus_helpers.tokenizers2 import AnyTokenizer

tok = AnyTokenizer(my_hf_tokenizer)   # any object with .encode(text).tokens
tok.tokenize("hello world")
```
