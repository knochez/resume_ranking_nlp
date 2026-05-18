from collections.abc import Iterable
from pathlib import Path

import numpy as np

from src import config
from src.vectorizers.embeddings import embed_texts


def score_semantic(
    resume_texts: Iterable[str],
    jd_texts: Iterable[str],
    model_name: str = config.EMBEDDING_MODEL,
    cache_dir: Path = config.CACHE_DIR,
) -> np.ndarray:
    """Score resume/JD pairs with sentence embeddings + cosine similarity.

    Both embedding arrays are L2-normalized, so cosine similarity reduces to
    a dot product computed via einsum.

    Args:
        resume_texts: Cleaned resume texts.
        jd_texts: Cleaned JD texts (same length as resume_texts).
        model_name: SentenceTransformer model identifier.
        cache_dir: Directory for embedding cache files.

    Returns:
        1-D float32 array of cosine similarities, one per pair.
    """
    resume_emb = embed_texts(list(resume_texts), label="resume", model_name=model_name, cache_dir=cache_dir)
    jd_emb = embed_texts(list(jd_texts), label="jd", model_name=model_name, cache_dir=cache_dir)
    return np.einsum("ij,ij->i", resume_emb, jd_emb)
