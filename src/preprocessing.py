import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))
_URL_PATTERN = re.compile(r"http\S+|www\.\S+")
_NONALPHA = re.compile(r"[^a-z0-9+#./\- ]")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize raw text for downstream vectorization.

    Mirrors the preprocessing used in ``notebooks/01_ranking_experiments_colab.ipynb``
    so the src-backed pipeline reproduces its numbers exactly:

    - lowercase, drop URLs,
    - keep tech-y punctuation (``+ # . / -``) so ``c++``, ``.net``, ``ci/cd`` survive,
    - lemmatize, drop stopwords and tokens of length 1.

    Note: the ``len(t) > 1`` filter drops single-character tokens, including
    single-digit numbers like ``5`` or ``3``. This is a deliberate, user-directed
    deviation from CLAUDE.md §6 to stay faithful to notebook 01 (see §9 decisions
    log, 2026-05-18).

    Args:
        text: Raw text (resume body, JD body, etc.).

    Returns:
        Cleaned, lemmatized text as a single space-separated string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _NONALPHA.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    tokens = [_LEMMATIZER.lemmatize(t) for t in text.split() if t not in _STOP_WORDS and len(t) > 1]
    return " ".join(tokens)
