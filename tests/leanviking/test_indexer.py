import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from leanviking.indexer import Indexer


def _make_embedder(vec=None):
    """Return a mock Embedder that produces a deterministic 384-d vector."""
    embedder = MagicMock()
    embedder.embed.return_value = (vec if vec is not None else np.ones(384)).tolist()
    return embedder


def test_rebuild_and_search_nested(tmp_path):
    """rebuild_index scans nested dirs; search returns viking:// URIs."""
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts" / "src"
    abstracts_dir.mkdir(parents=True)
    (abstracts_dir / "main.py.l0.txt").write_text("Entry point of the application")

    embedder = _make_embedder()
    indexer = Indexer(project_root=str(tmp_path), embedder=embedder)
    indexer.rebuild_index()

    results = indexer.search("entry point")

    assert results == ["viking://src/main.py"]


def test_rebuild_multiple_nested_files(tmp_path):
    """IDs preserve full relative path."""
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts"
    (abstracts_dir / "src" / "core").mkdir(parents=True)
    (abstracts_dir / "src" / "core" / "auth.py.l0.txt").write_text("Handles authentication")
    (abstracts_dir / "src" / "utils.py.l0.txt").write_text("Utility helpers")

    # Give distinct vectors so results are deterministic
    call_count = [0]
    vecs = [np.zeros(384).tolist(), np.ones(384).tolist()]

    def side_effect(text):
        v = vecs[call_count[0] % 2]
        call_count[0] += 1
        return v

    embedder = MagicMock()
    embedder.embed.side_effect = side_effect

    indexer = Indexer(project_root=str(tmp_path), embedder=embedder)
    indexer.rebuild_index()

    results = indexer.search("anything", top_k=2)
    uris = set(results)
    assert "viking://src/core/auth.py" in uris
    assert "viking://src/utils.py" in uris


def test_search_empty_index_returns_empty_list(tmp_path):
    embedder = _make_embedder()
    indexer = Indexer(project_root=str(tmp_path), embedder=embedder)
    assert indexer.search("anything") == []


def test_rebuild_empty_abstracts_dir_is_noop(tmp_path):
    (tmp_path / ".ov_brain" / "abstracts").mkdir(parents=True)
    embedder = _make_embedder()
    indexer = Indexer(project_root=str(tmp_path), embedder=embedder)
    indexer.rebuild_index()  # must not raise
    assert indexer.search("anything") == []
