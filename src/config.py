from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).parent.parent

SEED: int = 42
DATA_PATH: Path = _PROJECT_ROOT / "datasets" / "resume.csv"
CACHE_DIR: Path = _PROJECT_ROOT / "data" / "processed"
FIGURES_DIR: Path = _PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR: Path = _PROJECT_ROOT / "outputs" / "metrics"

# ----- Embeddings (Approach 2) -----
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE: int = 64

# ----- TF-IDF (Approach 1) -----
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
TFIDF_MIN_DF: int = 2
TFIDF_MAX_DF: float = 0.95
TFIDF_MAX_FEATURES: int = 30_000

# ----- Cross-encoder (Approach 3) -----
CE_BASE_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_LEN: int = 384
CE_EPOCHS: int = 3
CE_BATCH_SIZE: int = 16
CE_LR: float = 2e-5
CE_OUTPUT_DIR: Path = _PROJECT_ROOT / "outputs" / "models" / "cross_encoder_resume_jd"

# ----- Ranking evaluation -----
TOP_K: int = 10
RELEVANCE_QUANTILE: float = 0.5
