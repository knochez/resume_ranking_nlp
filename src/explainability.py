import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def top_shared_terms(
    vectorizer: TfidfVectorizer,
    resume_vec: sp.csr_matrix,
    jd_vec: sp.csr_matrix,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-k TF-IDF terms shared by a resume and a JD.

    Multiplies the two TF-IDF vectors element-wise; the largest products
    correspond to terms that appear with high weight in both.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        resume_vec: 1×V sparse TF-IDF vector for one resume.
        jd_vec: 1×V sparse TF-IDF vector for one JD.
        k: Number of top terms to return.

    Returns:
        List of (term, contribution) tuples, sorted descending by contribution.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    overlap = np.asarray(resume_vec.multiply(jd_vec).todense()).flatten()
    top_idx = overlap.argsort()[::-1][:k]
    return [(feature_names[i], float(overlap[i])) for i in top_idx if overlap[i] > 0]
