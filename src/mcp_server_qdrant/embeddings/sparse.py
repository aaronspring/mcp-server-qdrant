import logging

logger = logging.getLogger(__name__)


def sparse_vector_name(model_name: str) -> str:
    """
    Return the sparse vector name FastEmbed assigns to a model in Qdrant.
    Mirrors the dense convention in FastEmbedProvider.get_vector_name().
    """
    return f"fast-{model_name.split('/')[-1].lower()}"


def resolve_sparse_model_name(vector_name: str) -> str | None:
    """
    Resolve the FastEmbed sparse model id backing a collection's sparse vector.

    The vector name stored in Qdrant is derived from the model id, so the mapping
    is recovered by matching every supported sparse model against that convention,
    e.g. "fast-bm42-all-minilm-l6-v2-attentions" ->
    "Qdrant/bm42-all-minilm-l6-v2-attentions".

    :param vector_name: The name of the sparse vector in the collection.
    :return: The FastEmbed model id, or None if no supported model matches.
    """
    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        logger.warning("fastembed is not installed, cannot resolve sparse model name")
        return None

    for description in SparseTextEmbedding.list_supported_models():
        model_name = description["model"]
        if sparse_vector_name(model_name) == vector_name:
            return model_name

    logger.warning(
        f"No supported FastEmbed sparse model matches vector name {vector_name!r}; "
        "set SPARSE_EMBEDDING_MODEL to the model id explicitly"
    )
    return None
