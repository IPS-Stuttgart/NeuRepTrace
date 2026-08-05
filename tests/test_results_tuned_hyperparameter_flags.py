from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neureptrace.results import read_time_decode_results


def _result_rows(flags: list[object]) -> pd.DataFrame:
    n_rows = len(flags)
    return pd.DataFrame(
        {
            "subject": ["sub-01"] * n_rows,
            "fold": list(range(n_rows)),
            "time": [0.1 * index for index in range(n_rows)],
            "accuracy": [0.6] * n_rows,
            "log_loss": [0.5] * n_rows,
            "brier": [0.3] * n_rows,
            "ece": [0.1] * n_rows,
            "tuned_hyperparameters": flags,
        }
    )


@pytest.mark.parametrize("invalid_flag", ["enabled", "disabled", 2, -1, 0.5])
def test_read_time_decode_results_rejects_invalid_tuned_hyperparameter_flags(
    tmp_path: Path,
    invalid_flag: object,
) -> None:
    csv_path = tmp_path / "results.csv"
    _result_rows([invalid_flag]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="tuned_hyperparameters.*explicit true/false"):
        read_time_decode_results([csv_path])


def test_read_time_decode_results_normalizes_explicit_tuned_hyperparameter_flags(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    flags: list[object] = ["on", "off", "1", "0", "yes", "no", "true", "false", "", None]
    _result_rows(flags).to_csv(csv_path, index=False)

    results = read_time_decode_results([csv_path])

    assert results["tuned_hyperparameters"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
