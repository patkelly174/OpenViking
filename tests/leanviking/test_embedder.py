from unittest.mock import MagicMock, patch
from leanviking.embedder import RemoteEmbedder, LocalEmbedder, get_embedder


def _mock_litellm_response(embedding):
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].__getitem__ = lambda self, key: embedding if key == "embedding" else None
    return mock_response


class TestRemoteEmbedder:
    def test_embed_returns_float_list(self):
        with patch("litellm.embedding", return_value=_mock_litellm_response([0.1, 0.2, 0.3])) as mock_embed:
            embedder = RemoteEmbedder()
            result = embedder.embed("hello world")

            mock_embed.assert_called_once_with(model="text-embedding-3-small", input=["hello world"])
            assert isinstance(result, list)
            assert result == [0.1, 0.2, 0.3]

    def test_embed_uses_custom_model(self):
        with patch("litellm.embedding", return_value=_mock_litellm_response([1.0])) as mock_embed:
            embedder = RemoteEmbedder(model="text-embedding-ada-002")
            embedder.embed("test")
            mock_embed.assert_called_once_with(model="text-embedding-ada-002", input=["test"])


class TestLocalEmbedder:
    def test_embed_returns_float_list(self):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([MagicMock(tolist=lambda: [0.5, 0.6, 0.7])])

        with patch("leanviking.embedder.LocalEmbedder.__init__", lambda self, model: None):
            embedder = LocalEmbedder.__new__(LocalEmbedder)
            embedder._model = mock_model

        result = embedder.embed("hello")
        assert isinstance(result, list)

    def test_embed_materialises_generator(self):
        """FastEmbed returns a generator — confirm we consume it correctly."""
        import numpy as np
        arr = MagicMock()
        arr.tolist.return_value = [1.0, 2.0, 3.0]

        mock_model = MagicMock()
        mock_model.embed.return_value = (x for x in [arr])

        with patch("leanviking.embedder.LocalEmbedder.__init__", lambda self, model: None):
            embedder = LocalEmbedder.__new__(LocalEmbedder)
            embedder._model = mock_model

        result = embedder.embed("test")
        assert result == [1.0, 2.0, 3.0]


class TestGetEmbedder:
    def test_remote_mode_returns_remote_embedder(self):
        embedder = get_embedder(mode="remote")
        assert isinstance(embedder, RemoteEmbedder)

    def test_local_mode_returns_local_embedder(self):
        with patch("fastembed.TextEmbedding"):
            embedder = get_embedder(mode="local")
            assert isinstance(embedder, LocalEmbedder)

    def test_custom_model_passed_through(self):
        embedder = get_embedder(mode="remote", model="text-embedding-ada-002")
        assert embedder.model == "text-embedding-ada-002"
