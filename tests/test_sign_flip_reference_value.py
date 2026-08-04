from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.inference import sign_flip_time_inference, subject_time_effects


@pytest.mark.parametrize("entrypoint", [subject_time_effects, sign_flip_time_inference])
@pytest.mark.parametrize(
    "chance",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.asarray(True),
        np.asarray([0.5]),
        0.5 + 1.0j,
    ],
)
def test_inference_rejects_invalid_reference_values_before_file_access(
    tmp_path: Path,
    entrypoint: Callable,
    chance: object,
) -> None:
    missing_csv = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="chance must be a finite numeric scalar"):
        entrypoint([missing_csv], chance=chance)

    assert not missing_csv.exists()


def test_subject_time_effects_accepts_numpy_reference_scalar(tmp_path: Path) -> None:
    csv_path = tmp_path / "sub-01_time_decode.csv"
    pd.DataFrame(
        {
            "subject": ["sub-01"],
            "fold": [0],
            "time": [0.1],
            "accuracy": [0.7],
            "log_loss": [0.5],
            "brier": [0.2],
            "ece": [0.1],
        }
    ).to_csv(csv_path, index=False)

    effects = subject_time_effects([csv_path], chance=np.float64(0.5))

    assert effects.loc["sub-01", 0.1] == pytest.approx(0.2)
