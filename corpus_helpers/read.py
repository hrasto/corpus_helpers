import logging
from functools import partial
import re
from copy import copy

# --- text reading utils ---

def lower(texts):
    for text in texts:
        text = text.lower()
        yield text

def read(files, size = -1):
    for file in files:
        try:
            with open(file, 'r') as f:
                content = f.read(size)
            yield content
        except FileNotFoundError:
            logging.warning(f"file not found: {file}")

def split_by_fn(texts, divide_fn):
    for text in texts:
        for chunk in divide_fn(text):
            if not chunk: continue
            yield chunk

def split_by_regex(texts, pattern):
    return split_by_fn(texts, lambda text: re.split(pattern, text, flags=re.UNICODE))

# split_word = partial(split_by, pattern=r"[^\W\d_]+|\d+|[^\w\s]+")
split_word = partial(split_by_regex, pattern=r"(\W)")
split_line = partial(split_by_regex, pattern="(\n)")
split_pipe = partial(split_by_regex, pattern="(\\|)")

def as_bytes(texts, encoding='utf-8', errors='replace'):
    for text in texts:
        yield text.encode(encoding=encoding, errors=errors)

def append(texts, is_appendable=lambda s: not str.isalpha(s) and len(s)==1):
    buffer = next(texts)
    for text in texts:
        if is_appendable(text):
            buffer += text
        else:
            yield buffer
            buffer = text
    if buffer:
        yield buffer

def delete(texts, string=""):
    for text in texts:
        if text != string:
            yield text


delete_pipe = partial(delete, string="|")
delete_newline = partial(delete, string="\n")
delete_blank = partial(delete, string=' ')


def load_lexicon(path) -> dict[str, list[str]]:
    """Load a pipe-delimited morpheme lexicon (one word per line, e.g. 'walk|ing')."""
    lexicon = {}
    lines = restartable_file_reader([path], preprocessors=[split_line, delete_newline])
    for line in lines: 
        segs = line.split('|')
        if segs:
            lexicon["".join(segs)] = segs
    return lexicon


class _RestartableChain:
    def __init__(self, texts, preprocessors):
        self.texts = texts
        self.preprocessors = preprocessors
        self.length = None

    def __iter__(self):
        gen = copy(self.texts)
        for pp in self.preprocessors:
            gen = pp(gen)
        return gen
    
    def __len__(self):
        if self.length is None: 
            self.length = sum(1 for _ in self)
        return self.length


class _RestartableReader:
    def __init__(self, files, preprocessors, size):
        self.files = files
        self.preprocessors = preprocessors
        self.size = size

    def __iter__(self):
        texts = read(self.files, self.size)
        if self.preprocessors:
            for pp in self.preprocessors:
                texts = pp(texts)
        return texts


def chain_preprocessors(texts, preprocessors):
    return _RestartableChain(texts, preprocessors)

def restartable_file_reader(files, preprocessors=None, size=-1):
    return _RestartableReader(files, preprocessors, size)


class GroupView:
    """Wraps a flat or nested list of filenames into a lazy iterable of restartable readers.

    flat list [f1, f2, ...]           -> doc mode: yields one restartable reader per file
    nested list [[f1, f2], [f3], ...] -> group mode: yields one restartable reader per group,
                                         with all group files chained into a single stream

    Mode is inferred from the structure of `files`. In both modes each yielded item is a
    restartable iterable, so vocab_builder always receives the same type regardless of mode.
    """
    def __init__(self, files, preprocessors=None, size=-1):
        self._group_mode = bool(files) and not isinstance(files[0], str)
        self._files = files
        self._preprocessors = preprocessors
        self._size = size

    def __len__(self):
        return len(self._files)

    def __iter__(self):
        if self._group_mode:
            for group_files in self._files:
                yield restartable_file_reader(group_files, self._preprocessors, self._size)
        else:
            for f in self._files:
                yield restartable_file_reader([f], self._preprocessors, self._size)
