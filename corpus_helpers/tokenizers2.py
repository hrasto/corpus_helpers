"""
Tokenizers. Mostly wrappers over HF tokenizers, with a few twists. 
All methods should work with byte-strings instead of unicode character strings. 

Name: didn't want to conflate huggingface tokenizers and this tokenizers, so named it tokenizers2. 
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable
from tokenizers import Tokenizer
import json
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import ByteLevel
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

byte_encoder = _make_byte_encoder_dict()

def to_bytelevel(text: str) -> str:
    return "".join(byte_encoder[b] for b in text.encode("utf-8"))

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
    def encode(self, text: str) -> list[str]:
        """Tokenize text and return token strings."""
        ...

    def encode_batch(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]
    
    def tokenize(self, text: str) -> list[str]:
        return self.encode(text)
    
    def tokenize_batch(self, texts: Iterable[str]) -> list[list[str]]:
        return self.encode_batch(texts)
    
    def is_trained(self): 
        return self.tokenizer is not None

class AnyTokenizer(BaseTokenizer): 
    """ not sure if this is useful but this just takes an 'external' tokenizer object that has a tokenize member function and calls it instead of using its own implementation """
    def __init__(self, tokenizer,  name=None):
        super().__init__(None, tokenizer.get_vocab_size(), dont_train=True)
        self.tokenizer = tokenizer
        self.name = name
    def train(self, texts, **kwargs):
        pass
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

class LeftmostLongestTokenizer(BaseTokenizer):
    """ greedy tokenizer a.k.a maximum-matching tokenizer (uses HF's fallback implementation, when BPE is not supplied with a merge list) """
    def __init__(self, terms, pre_tokenizer=None, normalizer=None, **kwargs):
        if isinstance(terms, dict): 
            terms = list(terms.keys())
        if not isinstance(terms, list): 
            terms = list(terms)
        self.terms = terms
        self.pre_tokenizer = pre_tokenizer
        self.normalizer = normalizer 
        super().__init__(texts=None, vocab_size=len(terms), dont_train=False, **kwargs)

    def train(self, texts, **kwargs):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(self.terms) # the HF tokenizer defaults to the leftmost-longest method when initialized in this way
        tokenizer.pre_tokenizer = self.pre_tokenizer
        tokenizer.normalizer = self.normalizer
        self.tokenizer = tokenizer
    
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens
    
class ExactTokenizer(BaseTokenizer): 
    """a tokenizer which finds the exact best tokenization, based on a criterion (e.g. minimize the number of tokens), over a direct acyclic graph (DAC) """
    pass

# ---------- wrappers over standard HF tokenizers, probably not useful ---------

class BPETokenizer(BaseTokenizer):
    def __init__(self, texts, vocab_size, byte_level=True, dont_train=False, **kwargs):
        self.byte_level=byte_level
        super().__init__(texts, vocab_size, dont_train=False, **kwargs)

    def train(self, texts, **kwargs):
        trainer = BpeTrainer(vocab_size=self.vocab_size, initial_alphabet=ByteLevel.alphabet(), **kwargs)
        bpe = Tokenizer(BPE())
        if self.byte_level: 
            # uni.pre_tokenizer = ByteLevel(add_prefix_space=False)
            texts = map(to_bytelevel, texts)
        bpe.train_from_iterator(texts, trainer)
        self.tokenizer = bpe

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

class UnigramTokenizer(BaseTokenizer): 
    def __init__(self, texts, vocab_size, byte_level=True, dont_train=False, **kwargs):
        self.byte_level=byte_level
        super().__init__(texts, vocab_size, dont_train, **kwargs)

    def train(self, texts, **kwargs):
        trainer = UnigramTrainer(vocab_size=self.vocab_size, 
                                 initial_alphabet=ByteLevel.alphabet(),
                                 **kwargs)
        uni = Tokenizer(Unigram())
        if self.byte_level: 
            # uni.pre_tokenizer = ByteLevel(add_prefix_space=False)
            texts = map(to_bytelevel, texts)
        uni.train_from_iterator(texts, trainer)
        self.tokenizer = uni

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens