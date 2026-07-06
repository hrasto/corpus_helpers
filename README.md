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
Each corpus is an **iterable of `bytes`** objects (documents or lines), e.g. `[line.encode() for line in open(f)]`.

### N-gram overlap and divergence

```python
from corpus_helpers.metrics import ngram_overlap, ngram_divergence

a = [b"the cat sat on the mat", b"a quick brown fox"]
b = [b"the dog lay on the rug", b"a lazy brown dog"]

# Jaccard overlap of n-gram vocabularies (0 = disjoint, 1 = identical)
ngram_overlap(a, b)                          # word unigrams (default)
ngram_overlap(a, b, n=2, unit="char")        # character bigrams

# Jensen-Shannon divergence between n-gram frequency distributions (0 = identical)
ngram_divergence(a, b)                       # word unigrams, JSD
ngram_divergence(a, b, method="kl")          # KL divergence (asymmetric)
ngram_divergence(a, b, n=3, unit="char")     # character trigrams
ngram_divergence(a, b, smoothing=0.5)        # custom Laplace smoothing pseudocount
```

`unit="word"` tokenises on Unicode word boundaries; `unit="char"` splits into individual characters. `smoothing` controls the additive (Laplace) pseudocount added to every vocabulary entry before normalisation (default `1.0`).

### Normalized Compression Distance

```python
from corpus_helpers.metrics import (
    normalized_compression_distance,
    normalized_compression_distance_asymmetric,
)

# Symmetric NCD — concatenates documents, compresses with zlib
ncd = normalized_compression_distance(a, b)
# → float near 0 for similar corpora, near 1 for unrelated ones
# symmetric=True averages C(ab) and C(ba) to remove order-dependence
ncd = normalized_compression_distance(a, b, symmetric=True)
# Pass a custom compressor (e.g. bz2.compress, lzma.compress) as the third argument

# Asymmetric NCD — suited for |a| >> |b|
# Returns (C(ab) − C(a)) / C(b): fraction of b's information not captured in a
ncd_asym = normalized_compression_distance_asymmetric(a, b)
```

### BPE merge-rank correlation

```python
from corpus_helpers.metrics import bpe_merge_rank_correlation

score = bpe_merge_rank_correlation(a, b, vocab_size=8000, method="kendall")
# → float in [-1, 1]; higher means more similar BPE merge structure
# method="spearman" is also available; Kendall's τ is more robust to heavy-tailed rank distributions
```

### BPE overlap

`bpe_overlap` is an **asymmetric** metric that measures how well BPE merge rules learned from corpus `a` compress corpus `b`. It is more sensitive than merge-rank correlation because it evaluates actual tokenisation efficiency rather than rank agreement.

```python
from corpus_helpers.metrics import bpe_overlap

result = bpe_overlap(a, b)
# result["auc"]          — area under the BPT_rel curve ∈ [0, 1]; higher = more overlap
# result["bpt_skyline"]  — bytes-per-token when BPE is trained *on* b (upper bound for b)
# result["curve"]        — list of (k, bpt_rel) pairs; bpt_rel=1 means as efficient as skyline
# result["n_merges"]     — number of merge rules learned from a
```

The curve traces BPT_rel(k) as the top-k merge rules from `a` are applied to `b`, with k log-spaced from 1 to `n_merges`. The AUC summarises the whole curve in one number.

Key parameters:

| Parameter | Default | Effect |
|---|---|---|
| `min_freq_fract` | `1e-7` | Merge rules with frequency below `min_freq_fract × len(a)` bytes are dropped. Higher = fewer, higher-quality merges; lower = more merges including rare ones. |
| `k_steps` | `20` | Number of points on the curve (log-spaced). |
| `vocab_size` | `1_000_000` | Hard cap on the number of merge rules trained. |

`bpe_overlap(a, b)` ≠ `bpe_overlap(b, a)` in general: if `a` has a much larger corpus, its merge table tends to generalise better to `b` than vice versa.

### Sampled distance

`sampled_distance` wraps any corpus-level metric with convergent random sampling, useful when corpora are large and a single full-corpus evaluation would be slow or memory-intensive.

```python
from corpus_helpers.metrics import sampled_distance, ngram_divergence
from functools import partial

metric = partial(ngram_divergence, method="jsd")
result = sampled_distance(a, b, metric, sample_size_per_iteration=100_000, seed=42)
# result["mean"]         — mean metric value across all iterations
# result["std"]          — std of the last k values (convergence window)
# result["n_iterations"] — how many iterations were run
# result["converged"]    — True if std dropped below threshold before max_iterations
```

Each iteration draws a random subset of documents whose total byte size is at least `sample_size_per_iteration`, evaluates `metric`, and checks whether the rolling standard deviation of the last `k` values has fallen below `threshold`. Set `return_window=True` to get all per-iteration values in `result["values"]`.

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

### BPE tokenizer (trained from scratch)

```python
from corpus_helpers.tokenizers2 import BPETokenizer

tok = BPETokenizer(texts, vocab_size=8000)
tok.tokenize("hello world")   # → list of token strings
```

Trains a standard byte-level BPE tokenizer via HuggingFace `tokenizers`. Any keyword arguments accepted by `BpeTrainer` (e.g. `min_frequency`) can be passed through.

### BPE tokenizer from a merge list

`BPEFromMerges` constructs a BPE tokenizer from a pre-existing list of merge rules, and supports incremental extension without retraining.

```python
from corpus_helpers.tokenizers2 import BPEFromMerges

merges = [("h", "e"), ("he", "l"), ...]   # list of (str, str) pairs

tok = BPEFromMerges(merges)
tok.tokenize("hello")              # → list of token strings

# Extend incrementally — only processes the new merges
tok.extend([("hel", "lo")])

# Measure mean bytes-per-token over a corpus (list[bytes])
bpt = tok.measure_bpt(corpus)     # → float
```

The internal vocab is maintained incrementally in Python, so `extend()` is O(new merges). The underlying HF `Tokenizer` object is rebuilt on each `extend()` call (HF has no public add-merge API), but that cost is unavoidable. This class is used internally by `bpe_overlap` to sweep k values efficiently.

### Wrapping an external tokenizer

```python
from corpus_helpers.tokenizers2 import AnyTokenizer

tok = AnyTokenizer(my_hf_tokenizer)   # any object with .encode(text).tokens
tok.tokenize("hello world")
```
