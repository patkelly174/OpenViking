import pytest
from pathlib import Path
from src.cli import _read_source_file

def test_read_source_file_returns_content():
    test_file = Path("test_breadcrumbs_source.py")
    content = "import os\nprint('hello')\n# comment"
    test_file.write_text(content)

    try:
        result = _read_source_file(test_file)
        assert result == content
    finally:
        if test_file.exists():
            test_file.unlink()

def test_read_source_file_empty():
    test_file = Path("test_empty.py")
    test_file.write_text("")

    try:
        result = _read_source_file(test_file)
        assert result == ""
    finally:
        if test_file.exists():
            test_file.unlink()

def test_read_source_file_binary_returns_none(tmp_path):
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02")
    assert _read_source_file(binary_file) is None
