from pathlib import Path

import pandas as pd
import pytest

from neureptrace.metadata import prepare_binary_metadata


def _write_events(path: Path) -> bytes:
    pd.DataFrame({"category": ["face", "chair"]}).to_csv(path, index=False)
    return path.read_bytes()


def _prepare(events_csv: Path, out_path: Path) -> None:
    prepare_binary_metadata(
        events_csv,
        out_path,
        source_column="category",
        positive_pattern="face",
        label_column="condition",
    )


def test_prepare_binary_metadata_rejects_source_as_output(tmp_path: Path):
    events_csv = tmp_path / "events.csv"
    original = _write_events(events_csv)

    with pytest.raises(ValueError, match="must not overwrite events_csv"):
        _prepare(events_csv, events_csv)

    assert events_csv.read_bytes() == original


def test_prepare_binary_metadata_rejects_normalized_source_alias(tmp_path: Path):
    events_csv = tmp_path / "events.csv"
    original = _write_events(events_csv)
    aliased_output = tmp_path / "unused" / ".." / events_csv.name

    with pytest.raises(ValueError, match="must not overwrite events_csv"):
        _prepare(events_csv, aliased_output)

    assert events_csv.read_bytes() == original
