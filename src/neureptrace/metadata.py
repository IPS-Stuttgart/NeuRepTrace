from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Pattern

import pandas as pd


def _non_empty_text(value: str, *, name: str) -> str:
    text = str(value)
    if text.strip() == "":
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _compile_pattern(pattern: str, *, name: str, case_sensitive: bool) -> Pattern[str]:
    text = _non_empty_text(pattern, name=name)
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(text, flags=flags)
    except re.error as exc:
        raise ValueError(f"{name} must be a valid regular expression: {exc}.") from exc


def _validate_binary_labels(positive_label: str, negative_label: str) -> tuple[str, str]:
    positive = _non_empty_text(positive_label, name="positive_label")
    negative = _non_empty_text(negative_label, name="negative_label")
    if positive == negative:
        raise ValueError("positive_label and negative_label must be distinct.")
    return positive, negative


def add_binary_label(
    metadata: pd.DataFrame,
    *,
    source_column: str,
    positive_pattern: str,
    label_column: str,
    negative_pattern: str | None = None,
    positive_label: str = "positive",
    negative_label: str = "negative",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Add a binary label column by matching text patterns in an existing column.

    When ``negative_pattern`` is omitted, every non-null source value that does
    not match ``positive_pattern`` receives the negative label. When
    ``negative_pattern`` is provided, unmatched rows receive missing labels and
    rows that match both patterns keep the positive label.
    """
    source_column = _non_empty_text(source_column, name="source_column")
    label_column = _non_empty_text(label_column, name="label_column")
    positive_regex = _compile_pattern(positive_pattern, name="positive_pattern", case_sensitive=case_sensitive)
    negative_regex = None if negative_pattern is None else _compile_pattern(negative_pattern, name="negative_pattern", case_sensitive=case_sensitive)
    positive_label, negative_label = _validate_binary_labels(positive_label, negative_label)

    if source_column not in metadata.columns:
        raise ValueError(f"Source column '{source_column}' not found in metadata.")
    if label_column in metadata.columns:
        raise ValueError(f"Label column '{label_column}' already exists.")

    source = metadata[source_column].astype("string")
    positive = source.str.contains(positive_regex, regex=True, na=False)
    if negative_regex is None:
        negative = source.notna() & ~positive
    else:
        negative = source.str.contains(negative_regex, regex=True, na=False) & ~positive

    labeled = metadata.copy()
    labeled[label_column] = pd.NA
    labeled.loc[positive, label_column] = positive_label
    labeled.loc[negative, label_column] = negative_label
    return labeled


def prepare_binary_metadata(
    events_csv: Path,
    out_path: Path,
    *,
    source_column: str,
    positive_pattern: str,
    label_column: str,
    negative_pattern: str | None = None,
    positive_label: str = "positive",
    negative_label: str = "negative",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Load metadata, add a binary label, and write the result as CSV."""
    metadata = pd.read_csv(events_csv)
    labeled = add_binary_label(
        metadata,
        source_column=source_column,
        positive_pattern=positive_pattern,
        negative_pattern=negative_pattern,
        label_column=label_column,
        positive_label=positive_label,
        negative_label=negative_label,
        case_sensitive=case_sensitive,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_path, index=False)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a binary decoding label to an events or metadata CSV."
    )
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-column", required=True)
    parser.add_argument("--positive-pattern", required=True)
    parser.add_argument("--negative-pattern")
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--positive-label", default="positive")
    parser.add_argument("--negative-label", default="negative")
    parser.add_argument("--case-sensitive", action="store_true")
    args = parser.parse_args()

    labeled = prepare_binary_metadata(
        events_csv=args.events_csv,
        out_path=args.out,
        source_column=args.source_column,
        positive_pattern=args.positive_pattern,
        negative_pattern=args.negative_pattern,
        label_column=args.label_column,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        case_sensitive=args.case_sensitive,
    )
    counts = labeled[args.label_column].value_counts(dropna=False).to_dict()
    print(f"Wrote {args.out}")
    print(f"Label counts: {counts}")


if __name__ == "__main__":
    main()
