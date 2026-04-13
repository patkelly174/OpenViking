import pytest
from unittest.mock import MagicMock, patch
from leanviking.reranker import Reranker


def _make_mock_reranker(scores):
    """Patch TextCrossEncoder to return fixed scores."""
    mock_encoder = MagicMock()
    mock_encoder.rerank.return_value = iter(scores)
    return mock_encoder


def test_rerank_returns_top_k_indices():
    with patch("leanviking.reranker.Reranker.__init__", lambda self, model=None: None):
        r = Reranker.__new__(Reranker)
        r._model = _make_mock_reranker([0.1, 0.9, 0.4])

    docs = ["doc_a", "doc_b", "doc_c"]
    indices = r.rerank("query", docs, top_k=2)

    assert indices == [1, 2]  # 0.9, 0.4 → indices 1, 2


def test_rerank_empty_docs():
    with patch("leanviking.reranker.Reranker.__init__", lambda self, model=None: None):
        r = Reranker.__new__(Reranker)
        r._model = MagicMock()

    result = r.rerank("query", [], top_k=3)
    assert result == []
    r._model.rerank.assert_not_called()


def test_rerank_top_k_clamped_to_doc_count():
    with patch("leanviking.reranker.Reranker.__init__", lambda self, model=None: None):
        r = Reranker.__new__(Reranker)
        r._model = _make_mock_reranker([0.3, 0.7])

    docs = ["a", "b"]
    indices = r.rerank("query", docs, top_k=10)
    # Should return only 2 indices even though top_k=10
    assert len(indices) == 2
