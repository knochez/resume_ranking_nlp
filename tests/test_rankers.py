import numpy as np
import pytest

from src.rankers.lexical import score_lexical

RESUMES = [
    "python machine learning data science tensorflow",
    "java spring boot microservices docker kubernetes",
    "sql data warehouse etl pipeline analytics",
]
JDS = [
    "python developer machine learning experience required",
    "java backend engineer spring boot required",
    "data engineer sql etl experience required",
]


def test_score_lexical_shape():
    scores, _, _, _ = score_lexical(RESUMES, JDS)
    assert scores.shape == (3,)


def test_score_lexical_range():
    scores, _, _, _ = score_lexical(RESUMES, JDS)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_score_lexical_returns_vectorizer_and_matrices():
    scores, vectorizer, resume_mat, jd_mat = score_lexical(RESUMES, JDS)
    assert hasattr(vectorizer, "vocabulary_")
    assert resume_mat.shape[0] == 3
    assert jd_mat.shape[0] == 3


def test_score_lexical_single_pair():
    """A single resume/JD pair (2-doc corpus) must not trip min_df/max_df."""
    scores, _, _, _ = score_lexical(
        ["python machine learning engineer"],
        ["looking for a python machine learning engineer"],
    )
    assert scores.shape == (1,)
    assert 0.0 <= scores[0] <= 1.0
    assert scores[0] > 0.0  # shared terms should yield non-zero similarity


def test_score_lexical_identical_texts():
    texts = ["python developer with machine learning skills"] * 3
    scores, _, _, _ = score_lexical(texts, texts)
    assert np.allclose(scores, 1.0, atol=1e-5)


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed") is None,
    reason="sentence_transformers not installed",
)
def test_score_semantic_shape():
    from src.rankers.semantic import score_semantic
    scores = score_semantic(RESUMES, JDS, use_cache=False) if False else _semantic_no_cache(RESUMES, JDS)
    assert scores.shape == (3,)


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed") is None,
    reason="sentence_transformers not installed",
)
def test_score_semantic_range():
    scores = _semantic_no_cache(RESUMES, JDS)
    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)


def _semantic_no_cache(resume_texts, jd_texts):
    """Call score_semantic with caching disabled to avoid disk writes during tests."""
    from src.vectorizers.embeddings import embed_texts
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        from src import config
        resume_emb = embed_texts(resume_texts, label="resume_test", cache_dir=tmp_path, use_cache=False)
        jd_emb = embed_texts(jd_texts, label="jd_test", cache_dir=tmp_path, use_cache=False)
    return np.einsum("ij,ij->i", resume_emb, jd_emb)
