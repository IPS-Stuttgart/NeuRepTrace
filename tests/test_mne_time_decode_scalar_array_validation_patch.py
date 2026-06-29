import numpy as np
import pytest

import neureptrace  # noqa: F401 - importing the package installs runtime validation patches.
from neureptrace.mne_time_decode import (
    _normalize_integer,
    _normalize_nonnegative_float,
    _normalize_positive_float,
    _normalize_positive_int,
    _normalize_pseudo_label_confidence_threshold,
    _normalize_unit_interval_float,
)


@pytest.mark.parametrize("value", [np.asarray(3), np.array([3]), np.asarray(True)])
def test_mne_time_decode_rejects_array_integer_scalar_controls(value):
    with pytest.raises(ValueError, match="pseudo_label_max_iterations must be an integer"):
        _normalize_positive_int(value, name="pseudo_label_max_iterations")

    with pytest.raises(ValueError, match="label_shuffle_seed must be an integer"):
        _normalize_integer(value, name="label_shuffle_seed")


@pytest.mark.parametrize("value", [np.asarray(0.25), np.array([0.25]), np.asarray(True)])
def test_mne_time_decode_rejects_array_float_scalar_controls(value):
    with pytest.raises(ValueError, match="dann_learning_rate must be positive and finite"):
        _normalize_positive_float(value, name="dann_learning_rate")

    with pytest.raises(ValueError, match="dann_weight_decay must be non-negative and finite"):
        _normalize_nonnegative_float(value, name="dann_weight_decay")

    with pytest.raises(ValueError, match=r"dann_dropout must be finite in \[0, 1\)"):
        _normalize_unit_interval_float(value, name="dann_dropout")

    with pytest.raises(ValueError, match="pseudo_label_confidence_threshold must be between 0 and 1"):
        _normalize_pseudo_label_confidence_threshold(value)
