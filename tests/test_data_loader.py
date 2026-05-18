import pandas as pd

from src.data_loader import build_composite_text, parse_list_string


def test_parse_list_string_valid():
    result = parse_list_string("['Python', 'Java', 'SQL']")
    assert result == ["Python", "Java", "SQL"]


def test_parse_list_string_already_list():
    result = parse_list_string(["A", "B"])
    assert result == ["A", "B"]


def test_parse_list_string_non_bracket_returns_single():
    # Mirrors notebook 01: a plain (non-bracket) string is wrapped as [s].
    result = parse_list_string("this is not a list")
    assert result == ["this is not a list"]


def test_parse_list_string_bracketed_malformed_fallback():
    # Bracketed-but-unparseable: strip brackets/quotes as a best effort.
    result = parse_list_string("['Python', 'SQL'")
    assert result == ["Python, SQL"]


def test_parse_list_string_nested_flattened():
    result = parse_list_string("[['Python', 'SQL'], ['Java']]")
    assert result == ["Python", "SQL", "Java"]


def test_parse_list_string_nan():
    result = parse_list_string(float("nan"))
    assert result == []


def test_parse_list_string_empty_string():
    assert parse_list_string("") == []


def test_build_composite_text_plain_columns():
    # Mirrors notebook 01: parts are joined with " . ".
    df = pd.DataFrame({"a": ["hello", "world"], "b": ["foo", "bar"]})
    result = build_composite_text(df, ["a", "b"])
    assert result.tolist() == ["hello . foo", "world . bar"]


def test_build_composite_text_list_columns():
    df = pd.DataFrame({"skills": ["['Python', 'SQL']", "['Java']"]})
    result = build_composite_text(df, ["skills"], list_fields=["skills"])
    assert "Python" in result[0]
    assert "SQL" in result[0]
    assert "Java" in result[1]


def test_build_composite_text_missing_column_skipped():
    df = pd.DataFrame({"a": ["hello"]})
    result = build_composite_text(df, ["a", "nonexistent"])
    assert result.tolist() == ["hello"]


def test_build_composite_text_empty_fields():
    df = pd.DataFrame({"a": ["hello"]})
    result = build_composite_text(df, [])
    assert result.tolist() == [""]
