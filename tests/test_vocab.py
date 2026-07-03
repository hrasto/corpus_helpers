from pathlib import Path
from collections import Counter
import pytest

from corpus_helpers.read import GroupView
from corpus_helpers.vocab import VocabFilter

FIXTURES = Path(__file__).parent / "fixtures"
DOC_A = str(FIXTURES / "doc_a.txt")  # "hello world\nfoo"
DOC_B = str(FIXTURES / "doc_b.txt")  # "hello bar\nbaz"
DOC_C = str(FIXTURES / "doc_c.txt")  # "world\nqux"


def word_counter(doc):
    """Simple vocab builder: splits each yielded string by whitespace."""
    return Counter(word for text in doc for word in text.split() if word)


# --- VocabFilter flat (doc-level df) ---

def test_vocabfilter_flat_term_frequency():
    vf = VocabFilter(GroupView([DOC_A, DOC_B, DOC_C]), word_counter)
    assert vf.tf["hello"] == 2
    assert vf.tf["world"] == 2
    assert vf.tf["foo"] == 1
    assert vf.tf["qux"] == 1

def test_vocabfilter_flat_document_frequency():
    vf = VocabFilter(GroupView([DOC_A, DOC_B, DOC_C]), word_counter)
    assert vf.df["hello"] == 2   # in doc_a and doc_b
    assert vf.df["world"] == 2   # in doc_a and doc_c
    assert vf.df["foo"] == 1     # only in doc_a
    assert vf.df["qux"] == 1     # only in doc_c

def test_vocabfilter_flat_len():
    vf = VocabFilter(GroupView([DOC_A, DOC_B, DOC_C]), word_counter)
    assert len(vf) == len(vf.tf)


# --- VocabFilter nested (group-level df) ---

def test_vocabfilter_nested_term_frequency():
    # tf is still per-token across all docs regardless of grouping
    vf = VocabFilter(GroupView([[DOC_A, DOC_B], [DOC_C]]), word_counter)
    assert vf.tf["hello"] == 2
    assert vf.tf["world"] == 2

def test_vocabfilter_nested_group_frequency():
    # group A = [doc_a, doc_b]: contains hello, world, foo, bar, baz
    # group B = [doc_c]:        contains world, qux
    vf = VocabFilter(GroupView([[DOC_A, DOC_B], [DOC_C]]), word_counter)
    assert vf.df["world"] == 2   # in both groups
    assert vf.df["hello"] == 1   # only in group A
    assert vf.df["qux"] == 1     # only in group B

def test_vocabfilter_nested_vs_flat_df_differs():
    flat = VocabFilter(GroupView([DOC_A, DOC_B, DOC_C]), word_counter)
    nested = VocabFilter(GroupView([[DOC_A, DOC_B], [DOC_C]]), word_counter)
    # "hello" appears in 2 docs but only 1 group
    assert flat.df["hello"] == 2
    assert nested.df["hello"] == 1


# --- VocabFilter set operations ---

def test_vocabfilter_intersection():
    vf_ab = VocabFilter(GroupView([DOC_A, DOC_B]), word_counter)
    vf_c = VocabFilter(GroupView([DOC_C]), word_counter)
    common = vf_ab.intersection(vf_c)
    assert "world" in common
    assert "hello" not in common

def test_vocabfilter_union():
    vf_ab = VocabFilter(GroupView([DOC_A, DOC_B]), word_counter)
    vf_c = VocabFilter(GroupView([DOC_C]), word_counter)
    all_terms = vf_ab.union(vf_c)
    assert all_terms == {"hello", "world", "foo", "bar", "baz", "qux"}

def test_vocabfilter_difference():
    vf_ab = VocabFilter(GroupView([DOC_A, DOC_B]), word_counter)
    vf_c = VocabFilter(GroupView([DOC_C]), word_counter)
    only_in_ab = vf_ab.difference(vf_c)
    assert "world" not in only_in_ab   # world is in both
    assert "hello" in only_in_ab
