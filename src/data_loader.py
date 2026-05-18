import ast
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Union

import pandas as pd


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Load the resume/JD CSV, tolerating encoding issues and a BOM header.

    Mirrors notebook 01 §5: read with ``encoding_errors="replace"`` and strip
    any BOM / surrounding whitespace from column names (the dataset ships with
    a BOM-prefixed ``job_position_name`` column).

    Args:
        path: Path to the CSV file.

    Returns:
        The loaded DataFrame with cleaned column names.
    """
    df = pd.read_csv(path, encoding_errors="replace")
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    return df


def parse_list_string(raw) -> list[str]:
    """Parse a stringified list like ``"['Python', 'Docker']"`` into a plain list.

    Mirrors notebook 01 §4.2. Uses ``ast.literal_eval`` (never ``eval``) and
    falls back gracefully: a bracketed-but-malformed string is stripped of its
    brackets/quotes, a non-bracket string is returned as a single-element list.

    Args:
        raw: Cell value that may be a stringified list, a real list, or NaN.

    Returns:
        A flat list of strings. Empty list on NaN/empty input.
    """
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str):
        # Scalar: NaN -> [], any other scalar -> its string form.
        return [] if pd.isna(raw) else [str(raw)]
    s = raw.strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            return [s.strip("[]").replace("'", "").replace('"', "")]
        flat: list[str] = []

        def _flatten(x):
            if isinstance(x, (list, tuple, set)):
                for item in x:
                    _flatten(item)
            elif x is not None:
                flat.append(str(x))

        _flatten(parsed)
        return flat
    return [s]


def build_composite_text(
    df: pd.DataFrame,
    fields: Sequence[str],
    list_fields: Iterable[str] = (),
) -> pd.Series:
    """Concatenate selected columns into one string per row.

    Mirrors notebook 01 §4.2: list-typed fields are parsed via
    :func:`parse_list_string` and space-joined; remaining fields are coerced to
    strings. Parts are joined with ``" . "`` and empty/NaN parts are skipped.

    Args:
        df: Source DataFrame.
        fields: Column names to include (in order).
        list_fields: Subset of ``fields`` whose values are stringified lists.

    Returns:
        A Series of length ``len(df)`` with the joined text per row.
    """
    list_fields_set = set(list_fields)

    def row_to_text(row):
        parts: list[str] = []
        for col in fields:
            if col not in row.index:
                continue
            val = row[col]
            if col in list_fields_set:
                parts.extend(parse_list_string(val))
            elif pd.notna(val):
                parts.append(str(val))
        return " . ".join(p for p in parts if p)

    return df.apply(row_to_text, axis=1)
