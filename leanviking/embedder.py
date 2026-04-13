import litellm

_DEFAULT_MODEL = "text-embedding-3-small"


class Embedder:
    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = litellm.embedding(model=self.model, input=[text])
        return response.data[0]["embedding"]
