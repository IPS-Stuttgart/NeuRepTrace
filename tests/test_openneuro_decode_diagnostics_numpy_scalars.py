from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401  # installs runtime compatibility patches
import neureptrace.openneuro_decode_diagnostics as diagnostics


def test_boolean_provenance_accepts_zero_dimensional_numpy_missing_value() -> None:
    value = np.array(np.nan)

    assert diagnostics._as_bool(value) is False
    assert diagnostics._optional_unique_bool(value, column="label_shuffle_control") is None


def test_boolean_provenance_accepts_zero_dimensional_numpy_boolean() -> None:
    value = np.array(True)

    assert diagnostics._as_bool(value) is True
    assert diagnostics._optional_unique_bool(value, column="label_shuffle_control") is True
