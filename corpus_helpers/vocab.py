"""Vocabulary building algorithms."""

from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.models import BPE, Unigram
from tokenizers import Tokenizer
from collections import Counter
from tokenizers.pre_tokenizers import ByteLevel
from .tokenizers2 import to_bytelevel
from picky_bpe.bpe_trainer import BPE as picky
import json
import morfessor
from tqdm import tqdm

def build_bpe_vocab(texts, max_token_length=20, vocab_size=1000000, byte_level=True) -> Counter: 
    tokenizer = Tokenizer(BPE())
    if byte_level: 
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bpe_trainer = BpeTrainer(vocab_size=vocab_size+4, # somehow, HF always reserves 4 entries for special tokens, even if we don't want any
                                show_progress=False, 
                                initial_alphabet=ByteLevel.alphabet(), 
                                min_frequency=2,
                                max_token_length=max_token_length)
    tokenizer.train_from_iterator(texts, bpe_trainer)
    merges = json.loads(tokenizer.to_str())["model"]["merges"]
    res_without_alphabet = Counter({s1+s2: freq for s1, s2, freq in merges})
    # add 1 to every token count as well (the alphabet-counter below will also add a finctional 1 to every letter)
    res_without_alphabet = Counter({word: count+1 for word, count in res_without_alphabet.most_common()})
    res = res_without_alphabet + Counter(ByteLevel.alphabet())
    return res

def build_picky_bpe_vocab(texts, threshold=.6, coverage=1.0, vocab_size=100000, byte_level=True) -> Counter: 
    # note that: (1) texts will be turned into a list and thus loaded into memory in full extent at once
    #            (2) picky bpe splits (pretokenizes) by whitespace
    #            (3) picky bpe prepends weird underscores: we remove them, and therefore may lose some types in this way; therefore we initially set the vocab size to a larger-than-desired value, and then take the top vocab-size entries
    trainer = picky(vocab_size=vocab_size*1.2, threshold=threshold, coverage=coverage)
    if byte_level:
        texts = list(map(to_bytelevel, texts))
    if not isinstance(texts, list):
        texts = list(texts)
    vocab, freqs = trainer.fit_return('\n'.join(texts))
    vocab = list(map(lambda s: s.replace('▁', ''), vocab))
    res_without_alphabet = {token: freq + 1 for token, freq in zip(vocab, freqs)}
    res_without_alphabet = Counter(res_without_alphabet)
    res_without_alphabet = Counter(dict(res_without_alphabet.most_common(vocab_size))) # truncate because of point (3)
    res = res_without_alphabet + Counter(ByteLevel.alphabet())
    return res

def build_unigram_vocab(texts, vocab_size=100000, max_token_length=20, byte_level=True) -> Counter: 
    tokenizer = Tokenizer(Unigram())
    if byte_level:
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = UnigramTrainer(vocab_size=vocab_size, 
                                        show_progress=False, 
                                        initial_alphabet=ByteLevel.alphabet(), 
                                        max_piece_length=max_token_length)
    tokenizer.train_from_iterator(texts, trainer)
    # load and encode files to get token frequencies
    decoded_tokens = [tok for text in texts for tok in tokenizer.encode(text).tokens]
    res = Counter(decoded_tokens)
    for char in ByteLevel.alphabet(): 
        if char not in res: 
            res[char] = 1
    return res

def build_morfessor_vocab(texts, corpusweight=1.0, vocab_size=None, byte_level=True):
    """
    Notes:
    (1) vocab size argument has no effect, it is there only so that all vocab builders have it as a parameter
    (2) morfessor always splits on whitespace
    """
    word_counts = Counter((to_bytelevel(word) if byte_level else word) for text in texts for word in text.split())

    model = morfessor.BaselineModel(corpusweight=corpusweight)
    model.load_data([(count, word) for word, count in word_counts.items()])
    model.train_batch()

    morphs = Counter(
        morph
        for count, compound, segmentation in model.get_segmentations()
        for morph in segmentation
        for _ in range(count)
    )

    # merge with byte-level alphabet for full coverage
    vocab = morphs | Counter(ByteLevel.alphabet())
    return vocab

# --- filtering methods ---

def _get_vocab_with_stats(docs, vocab_builder, total_docs=None):
    """can be used to create filtered vocabs with VocabFilter

    Args:
        docs (iterable): iterable of documents
        vocab_builder (callable): function that returns a Counter object (the values will be used as secondary ranking)

    Returns:
        tuple: (term_frequency:Counter, document_frequency:Counter)
    """
    doc_freq = Counter()
    term_freq = Counter()
    for doc in tqdm(docs, total=total_docs, disable=total_docs is None): 
        try: 
            vocab = vocab_builder(doc)
        except Exception as e: 
            print(f"Warning: vocabulary builder failed for file {doc[:200]} (error message: {str(e)})")
            continue
        term_freq.update(vocab)
        types, _ = zip(*vocab.items())
        doc_freq.update(types)
    return (term_freq, doc_freq)

class VocabFilter: 
    """ the class to be used for vocab filtering. it gathers the document-/group-frequencies of individual vocab entries, sorts the entries according to this statistic, and allows to filter (via a cut-off argument) the entries using the get_vocab method"""
    tf: Counter
    df: Counter

    def __init__(self, docs, vocab_builder, total_docs = None):
        self.tf, self.df = _get_vocab_with_stats(docs, vocab_builder, total_docs)

    def get_vocab(self, max_size=None, min_df=1, min_tf=1, order_by='df'):
        if order_by=='df' or order_by=='tf': 
            order_fn = lambda tf,df: tf if order_by=='tf' else df
        elif callable(order_by): 
            order_fn = order_by
        else: 
            raise ValueError('invalid order_by argument (must be "df", "tf", or a function taking a tf,df as arguments and returning a number)')
        counter_to_obide = {key: order_fn(self.tf[key], self.df[key]) for key in self.tf}
        counter_to_obide = Counter(counter_to_obide)
        ref_terms, ref_stat = zip(*counter_to_obide.most_common())
        terms_and_stats = [(term, stat) for term, stat in zip(ref_terms, ref_stat) if self._term_is_ok(term, min_df, min_tf)]
        if isinstance(max_size, int) and max_size > 0: 
            terms_and_stats = terms_and_stats[:max_size]
        terms, stats = zip(*terms_and_stats)

        missing = set(ByteLevel.alphabet()).difference(terms)
        if missing: terms = terms[:-len(missing)]
        # return the final set of terms and the last passing stat (criterion, e.g. document frequency)
        return set(terms).union(missing), stats[-1-len(missing)]
    
    def _term_is_ok(self, term, min_df, min_tf): 
        df_ok = term not in self.df or self.df[term] >= min_df
        tf_ok = term not in self.tf or self.tf[term] >= min_tf
        return df_ok and tf_ok
    
    def __len__(self): 
        return len(self.tf)
    
    def intersection(self, other): 
        return set(self.tf).intersection(other.tf)
    
    def union(self, other): 
        return set(self.tf).union(other.tf)
    
    def difference(self, other): 
        return set(self.tf).difference(other.tf)