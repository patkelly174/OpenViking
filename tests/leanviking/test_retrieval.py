import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from leanviking.brain import Brain


def test_get_context_flow(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/main.py"]

    l1_content = "L1 Content"

    with patch("pathlib.Path.read_text", return_value=l1_content):
        result = brain.get_context("query")

    assert result == l1_content
    brain.indexer.search.assert_called_once_with("query", top_k=3)


def test_get_context_uri_resolution(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/utils/helper.py"]

    with patch("pathlib.Path.read_text", return_value="Utils Overview"):
        result = brain.get_context("query")

    assert result == "Utils Overview"


def test_get_context_includes_l2_when_requested(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/main.py"]

    # Create the actual source file so L2 read succeeds
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "main.py").write_text("def main(): pass")

    # Create a fake L1 overview
    overviews_dir = tmp_path / ".ov_brain" / "overviews"
    overviews_dir.mkdir(parents=True)
    (overviews_dir / "src.l1.txt").write_text("L1 Overview")

    result = brain.get_context("query", include_l2=True)

    assert "L1 Overview" in result
    assert "def main(): pass" in result


def test_get_context_excludes_l2_by_default(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/main.py"]

    with patch("pathlib.Path.read_text", return_value="L1 Only"):
        result = brain.get_context("query")

    # Only one block — no L2
    assert result == "L1 Only"


def test_get_context_deduplicates_l1(tmp_path):
    """Multiple URIs from the same directory should return the L1 overview only once."""
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    # Three files in the same directory
    brain.indexer.search.return_value = [
        "viking://src/a.py",
        "viking://src/b.py",
        "viking://src/c.py",
    ]

    with patch("pathlib.Path.read_text", return_value="L1 Overview"):
        result = brain.get_context("query", top_k=3)

    # L1 should appear exactly once, not three times
    assert result.count("L1 Overview") == 1
