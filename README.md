# corpus_helpers: 

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