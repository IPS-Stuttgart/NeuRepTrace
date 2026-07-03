from pathlib import Path

import pandas as pd

from neureptrace.validate_manifest import _load_metadata_for_row


def test_validate_manifest_numeric_case_sensitive_flag_preserves_case(tmp_path: Path) -> None:
    events_csv = tmp_path / "events.csv"
    events_csv.write_text("stimulus\nCAT\ncat\n", encoding="utf-8")
    row = pd.Series(
        {
            "events_csv": "events.csv",
            "source_column": "stimulus",
            "positive_pattern": "CAT",
            "label_column": "condition",
            "case_sensitive": 1.0,
        }
    )

    metadata, messages = _load_metadata_for_row(row, tmp_path)

    assert messages == []
    assert metadata is not None
    assert metadata["condition"].tolist() == ["positive", "negative"]
