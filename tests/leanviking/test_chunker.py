import pytest
from leanviking.chunker import chunk_file, SymbolChunk


PYTHON_SOURCE = """\
def greet(name):
    return f"Hello, {name}"


class Greeter:
    def __init__(self, prefix):
        self.prefix = prefix

    def greet(self, name):
        return f"{self.prefix}, {name}"
"""

EMPTY_SOURCE = ""

JS_SOURCE = """\
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(a, b) {
        return a * b;
    }
}
"""


def test_python_extracts_functions_and_classes():
    chunks = chunk_file("src/greet.py", PYTHON_SOURCE)
    uris = [c.uri for c in chunks]
    anchors = [c.anchor for c in chunks]

    assert any("greet" in a for a in anchors)
    assert any("Greeter" in a for a in anchors)
    # All URIs start with the file path
    for uri in uris:
        assert uri.startswith("viking://src/greet.py")


def test_python_chunk_uris_have_anchors():
    chunks = chunk_file("src/greet.py", PYTHON_SOURCE)
    symbol_chunks = [c for c in chunks if c.anchor]
    assert len(symbol_chunks) >= 2  # at least greet + Greeter


def test_chunk_start_end_lines():
    chunks = chunk_file("src/greet.py", PYTHON_SOURCE)
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_empty_file_returns_single_whole_file_chunk():
    chunks = chunk_file("src/empty.py", EMPTY_SOURCE)
    assert len(chunks) == 1
    assert chunks[0].anchor == ""
    assert chunks[0].uri == "viking://src/empty.py"


def test_unsupported_extension_returns_single_chunk():
    chunks = chunk_file("config.yaml", "key: value\nother: 123\n")
    assert len(chunks) == 1
    assert chunks[0].anchor == ""
    assert "#" not in chunks[0].uri


def test_source_line_numbers_prefixed():
    """Chunks should have line numbers prepended for the summarizer."""
    chunks = chunk_file("src/greet.py", PYTHON_SOURCE)
    for chunk in chunks:
        if chunk.anchor:
            # Each line in the source should start with a line number
            first_line = chunk.source.splitlines()[0]
            assert first_line[0].isdigit(), f"Expected line number prefix, got: {first_line!r}"


def test_javascript_chunks():
    chunks = chunk_file("src/calc.js", JS_SOURCE)
    anchors = {c.anchor for c in chunks}
    assert any("add" in a for a in anchors)
    assert any("Calculator" in a for a in anchors)


def test_large_symbol_truncated():
    """Symbols exceeding _MAX_CHUNK_BYTES are truncated, not dropped."""
    from leanviking.chunker import _MAX_CHUNK_BYTES
    big_body = "    x = 1\n" * (_MAX_CHUNK_BYTES // 10 + 50)
    big_fn = f"def big_function():\n{big_body}"
    chunks = chunk_file("src/big.py", big_fn)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.source.encode()) <= _MAX_CHUNK_BYTES + 100  # small overhead OK
