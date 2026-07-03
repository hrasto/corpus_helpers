from pathlib import Path
from collections import Counter
import pytest

from corpus_helpers.read import (
    read,
    chain_preprocessors,
    restartable_file_reader,
    _RestartableReader,
    _RestartableChain,
    GroupView,
    lower,
    split_line,
)

FIXTURES = Path(__file__).parent / "fixtures"
DOC_A = str(FIXTURES / "doc_a.txt")
DOC_B = str(FIXTURES / "doc_b.txt")
DOC_C = str(FIXTURES / "doc_c.txt")


# --- read ---

def test_read_yields_file_contents():
    contents = list(read([DOC_A, DOC_B]))
    assert len(contents) == 2
    assert "hello" in contents[0]
    assert "bar" in contents[1]

def test_read_skips_missing_file():
    contents = list(read([DOC_A, "/nonexistent/file.txt"]))
    assert len(contents) == 1

def test_read_size_limits_bytes():
    contents = list(read([DOC_A], size=5))
    assert len(contents[0]) == 5


# --- _RestartableReader / restartable_file_reader ---

def test_restartable_reader_is_correct_type():
    reader = restartable_file_reader([DOC_A])
    assert isinstance(reader, _RestartableReader)

def test_restartable_reader_can_restart():
    reader = restartable_file_reader([DOC_A])
    assert list(reader) == list(reader)

def test_restartable_reader_yields_file_contents():
    reader = restartable_file_reader([DOC_A, DOC_B])
    contents = list(reader)
    assert len(contents) == 2
    assert "hello" in contents[0]

def test_restartable_reader_with_preprocessor():
    reader = restartable_file_reader([DOC_A], preprocessors=[lower])
    assert all(c == c.lower() for c in reader)

def test_restartable_reader_with_split_line():
    reader = restartable_file_reader([DOC_A], preprocessors=[split_line])
    lines = list(reader)
    assert len(lines) > 1

def test_restartable_reader_preprocessor_applied_on_restart():
    reader = restartable_file_reader([DOC_A], preprocessors=[lower])
    assert list(reader) == list(reader)


# --- _RestartableChain / chain_preprocessors ---

def test_restartable_chain_is_correct_type():
    reader = restartable_file_reader([DOC_A])
    chained = chain_preprocessors(reader, [lower])
    assert isinstance(chained, _RestartableChain)

def test_restartable_chain_applies_preprocessors_in_order():
    reader = restartable_file_reader([DOC_A], preprocessors=[split_line])
    chained = chain_preprocessors(reader, [lower])
    lines = list(chained)
    assert all(l == l.lower() for l in lines)

def test_restartable_chain_can_restart():
    reader = restartable_file_reader([DOC_A])
    chained = chain_preprocessors(reader, [lower, split_line])
    assert list(chained) == list(chained)


# --- GroupView ---

def test_groupview_flat_len():
    assert len(GroupView([DOC_A, DOC_B, DOC_C])) == 3

def test_groupview_nested_len():
    assert len(GroupView([[DOC_A, DOC_B], [DOC_C]])) == 2

def test_groupview_flat_yields_one_reader_per_file():
    gv = GroupView([DOC_A, DOC_B, DOC_C])
    items = list(gv)
    assert len(items) == 3
    assert all(isinstance(r, _RestartableReader) for r in items)

def test_groupview_flat_readers_have_correct_content():
    gv = GroupView([DOC_A, DOC_B])
    ra, rb = list(gv)
    assert "hello" in list(ra)[0]
    assert "bar" in list(rb)[0]

def test_groupview_nested_yields_one_reader_per_group():
    gv = GroupView([[DOC_A, DOC_B], [DOC_C]])
    items = list(gv)
    assert len(items) == 2
    assert all(isinstance(r, _RestartableReader) for r in items)

def test_groupview_nested_group_chains_files():
    gv = GroupView([[DOC_A, DOC_B], [DOC_C]])
    group_a, group_c = list(gv)
    assert len(list(group_a)) == 2
    assert len(list(group_c)) == 1

def test_groupview_readers_are_restartable():
    gv = GroupView([DOC_A, DOC_B])
    readers = list(gv)
    for r in readers:
        assert list(r) == list(r)

def test_groupview_with_preprocessors():
    gv = GroupView([DOC_A, DOC_B], preprocessors=[lower])
    for reader in gv:
        for text in reader:
            assert text == text.lower()
