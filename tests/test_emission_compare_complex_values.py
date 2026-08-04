from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.emission_compare import compare_emission_modes


def _temporal_summary() -> pd.DataFrame:
    rows = []
    for emission_mode in ("calibrated", "uncalibrated"):
        for condition, gain, p_value in (
            ("observed_effect", 0.12, None),
            ("baseline_window", 0.02, None),
            ("shuffled_time", 0.04, 0.02),
            ("shuffled_label", 0.03, 0.04),
        ):
            rows.append(
                {
                    "decoder": "linear_svm",
                    "emission_mode": emission_mode,
                    "condition": condition,
                    "persistence_gain_per_observation": gain,
                    "empirical_p_value": p_value,
                    "best_stay_probability": 0.9,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("column", "condition", "value"),
    [
        ("persistence_gain_per_observation", "observed_effect", 0.12 + 0.5j),
        ("empirical_p_value", "shuffled_time", np.complex128(0.02 + 0.1j)),
        ("best_stay_probability", "observed_effect", np.complex64(0.9 + 0.1j)),
    ],
)
def test_compare_emission_modes_rejects_complex_numeric_values(
    column: str,
    condition: str,
    value: complex,
) -> None:
    summary = _temporal_summary()
    summary[column] = summary[column].astype(object)
    row = summary.index[
        (summary["emission_mode"] == "calibrated")
        & (summary["condition"] == condition)
    ][0]
    summary.loc[row, column] = value

    with pytest.raises(ValueError, match=rf"{column} values must be real-valued"):
        compare_emission_modes(summary)
