from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.vectorizers.tfidf import tfidf_vectorize


def _per_row_cosine(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    """Row-wise cosine similarity between two sparse matrices of equal row count.

    Args:
        A: Sparse matrix of shape (n, vocab).
        B: Sparse matrix of shape (n, vocab).

    Returns:
        1-D array of n cosine similarity scores.
    """
    A_n = normalize(A, norm="l2", axis=1)
    B_n = normalize(B, norm="l2", axis=1)
    return np.asarray(A_n.multiply(B_n).sum(axis=1)).flatten()


def score_lexical(
    resume_texts: Iterable[str],
    jd_texts: Iterable[str],
) -> tuple[np.ndarray, TfidfVectorizer, sp.csr_matrix, sp.csr_matrix]:
    """Score resume/JD pairs with TF-IDF + cosine similarity.

    Args:
        resume_texts: Cleaned resume texts.
        jd_texts: Cleaned JD texts (same length as resume_texts).

    Returns:
        (scores, vectorizer, resume_matrix, jd_matrix) — scores is a 1-D
        float array of cosine similarities; the matrices and vectorizer are
        returned for downstream explainability use.
    """
    vectorizer, resume_matrix, jd_matrix = tfidf_vectorize(resume_texts, jd_texts)
    scores = _per_row_cosine(resume_matrix, jd_matrix)
    return scores, vectorizer, resume_matrix, jd_matrix
