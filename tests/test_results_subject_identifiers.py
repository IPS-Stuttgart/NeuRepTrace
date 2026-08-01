from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neureptrace.results import read_time_decode_results


def _result_rows(subject_column: str, invalid_subject: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            subject_column: ["sub-01", invalid_subject],
            "time": [0.1, 0.2],
            "accuracy": [0.6, 0.7],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.1, 0.2],
        }
    )


@pytest.mark.parametrize("invalid_subject", [None, "   "])
@pytest.mark.parametrize(
    ("csv_column", "subject_column"),
    [("subject", None), ("participant", "participant")],
)
def test_read_time_decode_results_rejects_missing_or_blank_subjects(
    tmp_path: Path,
    invalid_subject: object,
    csv_column: str,
    subject_column: str | None,
) -> None:
    csv_path = tmp_path / "results.csv"
    _result_rows(csv_column, invalid_subject).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="subject column.*non-missing, non-blank identifiers"):
        read_time_decode_results([csv_path], subject_column=subject_column)
