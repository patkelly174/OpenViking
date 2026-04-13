import litellm
from abc import ABC, abstractmethod
from typing import Optional

_DEFAULT_REMOTE_MODEL = "text-embedding-3-small"
_DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass


class RemoteEmbedder(BaseEmbedder):
    def __init__(self, model: str = _DEFAULT_REMOTE_MODEL):
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = litellm.embedding(model=self.model, input=[text])
        return response.data[0]["embedding"]


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model: str = _DEFAULT_LOCAL_MODEL):
        self.model_name = model
        # Lazy import: fastembed is heavy and downloads models on first use.
        # Users in remote mode should not pay this cost.
        from fastembed import TextEmbedding  # type: ignore[import-untyped]
        self._model = TextEmbedding(model=model)

    def embed(self, text: str) -> list[float]:
        # FastEmbed returns a generator of numpy arrays — must be materialised
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()


def get_embedder(mode: str = "remote", model: Optional[str] = None) -> BaseEmbedder:
    if mode == "local":
        return LocalEmbedder(model=model or _DEFAULT_LOCAL_MODEL)
    return RemoteEmbedder(model=model or _DEFAULT_REMOTE_MODEL)
