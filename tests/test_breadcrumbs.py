import pytest
from pathlib import Path
from leanviking.cli import _read_source_file

def test_read_source_file_line_numbering():
    """Verify that _read_source_file correctly prefixes lines with numbers."""
    # Create a dummy file
    test_file = Path("test_breadcrumbs_source.py")
    content = "import os\nprint('hello')\n# comment"
    test_file.write_text(content)

    try:
        numbered_content = _read_source_file(test_file)
        expected = "1: import os\n2: print('hello')\n3: # comment"
        assert numbered_content == expected
    finally:
        if test_file.exists():
            test_file.unlink()

def test_read_source_file_empty():
    """Verify that empty files are handled correctly."""
    test_file = Path("test_empty.py")
    test_file.write_text("")

    try:
        numbered_content = _read_source_file(test_file)
        assert numbered_content == ""
    finally:
        if test_file.exists():
            test_file.unlink()
