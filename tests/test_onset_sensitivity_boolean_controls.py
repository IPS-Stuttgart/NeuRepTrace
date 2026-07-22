from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neureptrace.onset_sensitivity import build_sensitivity_settings, run_onset_sensitivity


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_build_sensitivity_settings_rejects_non_boolean_stable_prediction_values(value: object) -> None:
    with pytest.raises(ValueError, match="stable_prediction_values must contain only boolean values"):
        build_sensitivity_settings(stable_prediction_values=(value,))


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_run_onset_sensitivity_rejects_non_boolean_stable_prediction_flag(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="include_stable_prediction must be a boolean value"):
        run_onset_sensitivity([], out_dir=tmp_path, include_stable_prediction=value)


def test_build_sensitivity_settings_accepts_numpy_boolean_values() -> None:
    settings = build_sensitivity_settings(
        stable_prediction_values=(np.bool_(False), np.bool_(True)),
    )

    assert [setting.require_stable_prediction for setting in settings] == [False, True]
