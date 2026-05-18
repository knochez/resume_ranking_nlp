import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from src import config


def _texts_hash(texts: list[str], model_name: str) -> str:
    """Produce a stable hash for a list of strings (for cache keying).

    Args:
        texts: Texts to hash.
        model_name: Model name included in the hash so different models get
            different cache files.

    Returns:
        16-character hex string.
    """
    h = hashlib.sha256()
    h.update(model_name.encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


def embed_texts(
    texts: Iterable[str],
    label: str,
    model_name: str = config.EMBEDDING_MODEL,
    cache_dir: Path = config.CACHE_DIR,
    use_cache: bool = True,
) -> np.ndarray:
    """Encode texts with a SentenceTransformer, caching to disk.

    Args:
        texts: Cleaned texts to encode.
        label: Short tag (e.g. "resume", "jd") used in cache filename.
        model_name: HuggingFace model identifier.
        cache_dir: Directory for `.npy` cache files.
        use_cache: If True, read/write cached embeddings.

    Returns:
        (n, d) float32 array of L2-normalized embeddings.
    """
    texts = list(texts)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"emb_{label}_{_texts_hash(texts, model_name)}.npy"

    if use_cache and cache_path.exists():
        print(f"  cache hit: {cache_path.name}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer  # lazy import

    print(f"  encoding {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if use_cache:
        np.save(cache_path, embeddings)
        print(f"  cached -> {cache_path.name}")
    return embeddings
