"""
tokenizers2. Mostly wrappers over HF tokenizers.
Not to conflate huggingface tokenizers and this tokenizers, named it tokenizers2. 

- dont use hf pre-tokenizers/normalizers, but supply iterables over whatever you want to tokenize. 
    - to pre-tokenize/normalize, preferably use iterators from corpus_helpers.read
- all tokenizers train/work on the byte level internally. this means that a string is first turned into bytes, and every byte is mapped to a displayable unicode character. some of these characters take up two bytes, so this is quite dirty. but it is how huggingface byte-level tokenizers do it though so we keep it the same. 
- individual tokens are therefore in the byte-level character set (unicode character boundaries are not enforced)
- to detokenize, the tokens need to be mapped to the byte strings that they represent, which are then concatenated, and decoded as utf-8
- detokenize(tokenize(string)) == string, so decoding errors should not occur
- the interface to the classes functions should be in the original string space, however
    - except for the output of methods that return individual tokens, like encode/encode_batch/get_merge_list -- these return strings in the byte-level character set
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable
from tokenizers import Tokenizer
import json
import logging
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.models import BPE, Unigram
import os
from collections import Counter
# from picky_bpe.bpe_trainer import BPE as picky
# import morfessor
from tqdm.auto import tqdm

# ------------- first a few utils ----------------------------------------------

def _make_byte_encoder_dict():
    """ for conversion to HF's ByteLevel alphabet with visible characters """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

_BYTE_TO_CHAR = _make_byte_encoder_dict()
_CHAR_TO_BYTE = {val: key for key, val in _BYTE_TO_CHAR.items()}
_ALPHABET = list(_CHAR_TO_BYTE.keys())

def to_bytelevel(text: str) -> str:
    return "".join(_BYTE_TO_CHAR[b] for b in text.encode("utf-8"))

def from_bytelevel(tokens: list[str]) -> str:
    text_bytes = bytes([_CHAR_TO_BYTE[ch] for token in tokens for ch in token])
    return text_bytes.decode(encoding="utf-8")

def _boundary_positions(segments: list[str]) -> set[int]:
    positions: set[int] = set()
    pos = 0
    for seg in segments[:-1]:
        pos += len(seg)
        positions.add(pos)
    return positions

# --------------- base tokenizer class and a few implementations ---------------

class BaseTokenizer(ABC):
    """
    Minimal interface every tokenizer must satisfy.

    encode() must return bare token strings (no special tokens, no padding).
    """

    vocab_size: int

    def __init__(self, texts: Iterable[str], vocab_size, dont_train=False, **kwargs):
        super().__init__()
        self.tokenizer = None
        self.vocab_size = vocab_size
        if not dont_train:
            self.train(texts=texts, **kwargs)

    @abstractmethod
    def train(self, texts: Iterable[str], **kwargs) -> None:
        """Train on a stream of plain-text lines. vocab_size must be honoured."""
        ...

    @abstractmethod
    def encode_str(self, text: str) -> list[str]:
        """Tokenize text and return token strings."""
        ...

    def decode_str(self, tokens: list[str]): 
        return from_bytelevel(tokens)
    
    def is_trained(self):
        return self.tokenizer is not None

    def measure_bpt(self, corpus: Iterable[str]) -> float:
        """Mean bytes-per-token over corpus.

        Corpus items are word tokens with heavy repetition, so we encode each
        unique item once and weight by its frequency. This is mathematically
        identical to encoding every doc, but cuts the number of HF encode()
        calls by ~50x — which also avoids the memory the Rust encode path
        retains and never returns to the OS.
        """
        freq = Counter(corpus)
        total_bytes = 0
        total_tokens = 0
        for doc, count in freq.items():
            total_bytes += count * len(doc.encode("utf-8"))
            total_tokens += count * len(self.encode_str(doc))
        return total_bytes / total_tokens if total_tokens > 0 else float("nan")

    def morphological_alignment(
        self,
        lexicon: dict[str, list[str]],
        min_segments: int = 1,
    ) -> dict[str, float]:
        """Boundary-F1 against a gold morpheme lexicon.

        lexicon maps surface form -> list of morpheme strings (original encoding,
        not byte-level). Use corpus_helpers.read.load_lexicon() to build it from a pipe-delimited file.
        """
        tp = fp = fn = 0
        for word, gold_segs in lexicon.items():
            if len(gold_segs) < min_segments:
                continue
            gold_segs_bl = [to_bytelevel(s) for s in gold_segs]
            pred_segs = self.encode_str(word)
            if "".join(pred_segs) != "".join(gold_segs_bl):
                logging.warning(
                    f"reconstruction mismatch: gold ({'|'.join(gold_segs_bl)}) "
                    f"!= pred ({'|'.join(pred_segs)})"
                )
                continue
            pred_b = _boundary_positions(pred_segs)
            gold_b = _boundary_positions(gold_segs_bl)
            tp += len(pred_b & gold_b)
            fp += len(pred_b - gold_b)
            fn += len(gold_b - pred_b)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return dict(precision=precision, recall=recall, f1=f1)

class AnyTokenizer(BaseTokenizer): 
    """ not sure if this is useful but this just takes an 'external' tokenizer object that has a tokenize member function and calls it instead of using its own implementation """
    def __init__(self, tokenizer,  name=None):
        super().__init__(None, tokenizer.get_vocab_size(), dont_train=True)
        self.tokenizer = tokenizer
        self.name = name
    def train(self, texts, **kwargs):
        pass
    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens

class LeftmostLongestTokenizer(BaseTokenizer):
    """ greedy tokenizer a.k.a maximum-matching tokenizer (uses HF's fallback BPE implementation, which is used when BPE is missing the merge list) """
    def __init__(self, terms, **kwargs):
        if isinstance(terms, dict): 
            terms = list(terms.keys())
        if not isinstance(terms, list): 
            terms = list(terms)
        self.terms = list(map(to_bytelevel, terms))
        super().__init__(texts=None, vocab_size=len(terms), dont_train=False, **kwargs)

    def train(self, texts: Iterable[str], **kwargs):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(self.terms) # the HF tokenizer defaults to the leftmost-longest method when initialized in this way
        self.tokenizer = tokenizer
    
    def encode_str(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens
    
class ExactTokenizer(BaseTokenizer): 
    """a tokenizer which finds the exact best tokenization, based on a criterion (e.g. minimize the number of tokens), over a direct acyclic graph (DAC) """
    pass

class BPETokenizer(BaseTokenizer):
    def __init__(self, texts: Iterable[str], vocab_size, dont_train=False, **kwargs):
        super().__init__(texts, vocab_size, dont_train=False, **kwargs)

    def train(self, texts, **kwargs):
        trainer = BpeTrainer(vocab_size=self.vocab_size, 
                             initial_alphabet=_ALPHABET, **kwargs)
        bpe = Tokenizer(BPE())
        bpe.train_from_iterator(map(to_bytelevel, texts), trainer)
        self.tokenizer = bpe

    def encode_str(self, text: str) -> list[str]:
        return self.tokenizer.encode(to_bytelevel(text)).tokens

    def get_merge_list(self):
        merges = json.loads(self.tokenizer.to_str())["model"]["merges"]
        # return [tuple(m.split(" ", 1)) for m in merges]
        return merges

class BPEFromMerges(BaseTokenizer):
    """BPE tokenizer constructed from a pre-existing merge list.

    The vocab dict is maintained incrementally in Python, so extend() only
    processes the new merges rather than rebuilding from scratch. The underlying
    HF Tokenizer object is recreated on each extend() call (HF has no public
    add-merge API), but that cost is proportional to the total merge count and
    unavoidable.
    """

    def __init__(self, merges: list[tuple[str, str]] = ()):
        self._vocab: dict[str, int] = {c: i for i, c in enumerate(sorted(_ALPHABET))}
        self._merges: list[tuple[str, str]] = []
        super().__init__(None, len(self._vocab), dont_train=True)
        if merges:
            self.extend(list(merges))

    def train(self, texts, **kwargs) -> None:
        pass

    def extend(self, new_merges: list[tuple[str, str]]) -> None:
        """Add merges and rebuild the HF tokenizer. O(new_merges) Python-side work."""
        for a, b in new_merges:
            merged = a + b
            if merged not in self._vocab:
                self._vocab[merged] = len(self._vocab)
            self._merges.append((a, b))
        self.vocab_size = len(self._vocab)
        self.tokenizer = Tokenizer(BPE(vocab=self._vocab, merges=self._merges))

    def encode_str(self, text: str) -> list[str]:
        return self.tokenizer.encode(to_bytelevel(text)).tokens


class UnigramTokenizer(BaseTokenizer):
    def __init__(self, texts, vocab_size, dont_train=False, **kwargs):
        super().__init__(texts, vocab_size, dont_train, **kwargs)

    def train(self, texts, **kwargs):
        trainer = UnigramTrainer(vocab_size=self.vocab_size, 
                                 initial_alphabet=_ALPHABET,
                                 **kwargs)
        uni = Tokenizer(Unigram())
        texts = map(to_bytelevel, texts)
        uni.train_from_iterator(texts, trainer)
        self.tokenizer = uni

    def encode_str(self, text):
        return self.tokenizer.encode(to_bytelevel(text)).tokens