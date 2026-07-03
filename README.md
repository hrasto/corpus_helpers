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

*TODO*

---

## Partitioning

*TODO*

---

## Tokenizers

*TODO*

## Running tests

```bash
pip install -r requirements.txt
pip install pytest
pytest tests/
```