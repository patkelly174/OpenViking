import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.indexer import Indexer


def _make_embedder(vec=None):
    embedder = MagicMock()
    embedder.embed.return_value = (vec if vec is not None else np.ones(384)).tolist()
    return embedder


def _write_l0(path: Path, search_text: str, display_text: str, uri: str = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"search_text": search_text, "display_text": display_text}
    if uri:
        data["uri"] = uri
    path.write_text(json.dumps(data))


def test_rebuild_and_search_nested(tmp_path):
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts" / "src"
    abstracts_dir.mkdir(parents=True)
    _write_l0(
        abstracts_dir / "main.py.l0.txt",
        search_text="entry point application startup",
        display_text="Entry point of the application.",
        uri="contextflow://src/main.py",
    )

    indexer = Indexer(project_root=str(tmp_path), embedder=_make_embedder())
    indexer.rebuild_index()
    results = indexer.search("entry point")
    assert results == ["contextflow://src/main.py"]


def test_rebuild_multiple_nested_files(tmp_path):
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts"
    _write_l0(
        abstracts_dir / "src" / "core" / "auth.py.l0.txt",
        search_text="authentication login credentials",
        display_text="Handles authentication.",
        uri="contextflow://src/core/auth.py",
    )
    _write_l0(
        abstracts_dir / "src" / "utils.py.l0.txt",
        search_text="utility helpers misc",
        display_text="Utility helpers.",
        uri="contextflow://src/utils.py",
    )

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
    assert "contextflow://src/core/auth.py" in uris
    assert "contextflow://src/utils.py" in uris


def test_search_empty_index_returns_empty_list(tmp_path):
    indexer = Indexer(project_root=str(tmp_path), embedder=_make_embedder())
    assert indexer.search("anything") == []


def test_rebuild_empty_abstracts_dir_is_noop(tmp_path):
    (tmp_path / ".ov_brain" / "abstracts").mkdir(parents=True)
    indexer = Indexer(project_root=str(tmp_path), embedder=_make_embedder())
    indexer.rebuild_index()
    assert indexer.search("anything") == []


def test_rebuild_legacy_plain_text_l0(tmp_path):
    """Legacy plain-text L0 files (no JSON) are handled gracefully."""
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts"
    abstracts_dir.mkdir(parents=True)
    legacy = abstracts_dir / "old.py.l0.txt"
    legacy.write_text("This is a plain text summary with no JSON.")

    indexer = Indexer(project_root=str(tmp_path), embedder=_make_embedder())
    indexer.rebuild_index()
    results = indexer.search("plain text")
    assert len(results) == 1


def test_get_display_texts(tmp_path):
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts"
    abstracts_dir.mkdir(parents=True)
    _write_l0(
        abstracts_dir / "foo.py.l0.txt",
        search_text="foo search",
        display_text="Foo does bar.",
        uri="contextflow://foo.py",
    )

    indexer = Indexer(project_root=str(tmp_path), embedder=_make_embedder())
    indexer.rebuild_index()
    texts = indexer.get_display_texts(["contextflow://foo.py"])
    assert texts["contextflow://foo.py"] == "Foo does bar."


def test_search_with_reranker(tmp_path):
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts"
    abstracts_dir.mkdir(parents=True)

    for name, search, display, uri in [
        ("a.py", "auth login jwt", "Auth module.", "contextflow://a.py"),
        ("b.py", "database orm query", "DB module.", "contextflow://b.py"),
    ]:
        _write_l0(abstracts_dir / f"{name}.l0.txt", search, display, f"contextflow://{name}")

    # Reranker that always puts b.py first regardless of vector score
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [1, 0]  # b first, then a

    call_count = [0]
    vecs = [np.zeros(384).tolist(), np.ones(384).tolist()]

    def side_effect(text):
        v = vecs[call_count[0] % 2]
        call_count[0] += 1
        return v

    embedder = MagicMock()
    embedder.embed.side_effect = side_effect

    indexer = Indexer(project_root=str(tmp_path), embedder=embedder, reranker=mock_reranker)
    indexer.rebuild_index()
    results = indexer.search("database", top_k=2)
    assert results[0] == "contextflow://b.py"
    mock_reranker.rerank.assert_called_once()
