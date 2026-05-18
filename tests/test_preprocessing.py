from src.preprocessing import clean_text


def test_clean_text_lowercases():
    assert clean_text("Hello World") == clean_text("hello world")


def test_clean_text_removes_urls():
    result = clean_text("Visit https://example.com for more info")
    assert "http" not in result
    assert "example" not in result


def test_clean_text_removes_emails():
    result = clean_text("Contact me at user@example.com")
    assert "@" not in result


def test_clean_text_preserves_multidigit_numbers():
    # Mirrors notebook 01: the len>1 token filter keeps multi-digit numbers
    # but drops single-character tokens (incl. single-digit numbers).
    result = clean_text("15 years of experience with Python 3").split()
    assert "15" in result
    assert "3" not in result  # single digit dropped (len == 1)
    assert "year" in result  # lemmatized "years"


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_removes_stopwords():
    result = clean_text("the quick brown fox")
    assert "the" not in result.split()


def test_clean_text_returns_string():
    assert isinstance(clean_text("some text"), str)
