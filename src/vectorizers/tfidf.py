from collections.abc import Iterable

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)


def tfidf_vectorize(
    resume_texts: Iterable[str],
    jd_texts: Iterable[str],
) -> tuple[TfidfVectorizer, sp.csr_matrix, sp.csr_matrix]:
    """Fit a TF-IDF vectorizer on the union of resume + JD text, return matrices.

    Fitting on the union ensures both spaces share a vocabulary so cosine
    similarity is well-defined. `sublinear_tf=True` dampens dominant terms
    (1 + log(tf) instead of raw tf).

    Args:
        resume_texts: Cleaned resume texts.
        jd_texts: Cleaned JD texts.

    Returns:
        (vectorizer, resume_matrix, jd_matrix) — sparse matrices.
    """
    resume_texts = list(resume_texts)
    jd_texts = list(jd_texts)
    n_docs = len(resume_texts) + len(jd_texts)

    # The configured min_df/max_df are tuned for the full dataset. On a tiny
    # corpus (e.g. a single resume/JD pair in the app) min_df=2 with a
    # fractional max_df becomes mathematically impossible, so fall back to
    # keeping every term.
    min_df = TFIDF_MIN_DF if n_docs > TFIDF_MIN_DF else 1
    max_df = TFIDF_MAX_DF if n_docs > TFIDF_MIN_DF else 1.0

    def _make(_min_df, _max_df) -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=_min_df,
            max_df=_max_df,
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
        )

    corpus = resume_texts + jd_texts
    vectorizer = _make(min_df, max_df)
    try:
        vectorizer.fit(corpus)
    except ValueError:
        # Degenerate corpus (e.g. every doc shares the same vocabulary, as in
        # the app's single-pair case): max_df/min_df prune everything. Fall
        # back to keeping all terms. The full dataset never hits this path.
        vectorizer = _make(1, 1.0)
        vectorizer.fit(corpus)
    resume_matrix = vectorizer.transform(resume_texts)
    jd_matrix = vectorizer.transform(jd_texts)
    return vectorizer, resume_matrix, jd_matrix
