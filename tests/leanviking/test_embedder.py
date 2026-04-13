from unittest.mock import MagicMock, patch
from leanviking.embedder import Embedder


def test_embed_returns_float_list():
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].__getitem__ = lambda self, key: [0.1, 0.2, 0.3] if key == "embedding" else None

    with patch("litellm.embedding", return_value=mock_response) as mock_embed:
        embedder = Embedder()
        result = embedder.embed("hello world")

        mock_embed.assert_called_once_with(
            model="text-embedding-3-small", input=["hello world"]
        )
        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]


def test_embed_called_with_custom_model():
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].__getitem__ = lambda self, key: [1.0] if key == "embedding" else None

    with patch("litellm.embedding", return_value=mock_response) as mock_embed:
        embedder = Embedder(model="text-embedding-ada-002")
        embedder.embed("test")
        mock_embed.assert_called_once_with(
            model="text-embedding-ada-002", input=["test"]
        )
