from typing import Optional


class Reranker:
    """Cross-encoder reranker that re-scores a candidate set against a query.

    Over-fetching from the vector index then re-ranking with a cross-encoder
    yields substantially better precision than raw cosine similarity alone.
    """

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._model = TextCrossEncoder(model_name=model)

    def rerank(self, query: str, docs: list[str], top_k: int) -> list[int]:
        """Return indices into *docs* of the top_k highest-scoring entries."""
        if not docs:
            return []
        scores = list(self._model.rerank(query, docs))
        ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
        return ranked[:top_k]


_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
