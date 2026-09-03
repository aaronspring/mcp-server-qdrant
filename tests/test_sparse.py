from mcp_server_qdrant.embeddings.sparse import (
    resolve_sparse_model_name,
    sparse_vector_name,
)
from mcp_server_qdrant.settings import QdrantSettings


class TestSparseVectorName:
    def test_strips_namespace_and_lowercases(self):
        assert (
            sparse_vector_name("Qdrant/bm42-all-minilm-l6-v2-attentions")
            == "fast-bm42-all-minilm-l6-v2-attentions"
        )
        assert sparse_vector_name("Qdrant/bm25") == "fast-bm25"


class TestResolveSparseModelName:
    def test_resolves_supported_models(self):
        """The model id has to be recovered from the collection's vector name."""
        assert (
            resolve_sparse_model_name("fast-bm42-all-minilm-l6-v2-attentions")
            == "Qdrant/bm42-all-minilm-l6-v2-attentions"
        )
        assert resolve_sparse_model_name("fast-bm25") == "Qdrant/bm25"

    def test_roundtrip_for_every_supported_model(self):
        from fastembed import SparseTextEmbedding

        for description in SparseTextEmbedding.list_supported_models():
            model_name = description["model"]
            resolved = resolve_sparse_model_name(sparse_vector_name(model_name))
            assert sparse_vector_name(resolved) == sparse_vector_name(model_name)

    def test_returns_none_for_unknown_vector_name(self):
        """Unknown names must not fall back to a wrong model, so hybrid can degrade."""
        assert resolve_sparse_model_name("sparse") is None
        assert resolve_sparse_model_name("fast-not-a-real-model") is None


class TestSparseSettings:
    def test_sparse_embedding_model_defaults_to_none(self):
        assert QdrantSettings().sparse_embedding_model is None

    def test_sparse_embedding_model_from_env(self, monkeypatch):
        monkeypatch.setenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
        assert QdrantSettings().sparse_embedding_model == "Qdrant/bm25"
