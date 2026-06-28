from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace._bushmeg_all_protocols_prediction_metric_patch import _numeric_label_indices


def test_prediction_metric_label_indices_accept_exact_integer_values() -> None:
    labels = _numeric_label_indices(pd.Series([0, "1", 2.0]))

    assert labels is not None
    assert labels.dtype == np.dtype(int)
    assert labels.tolist() == [0, 1, 2]


def test_prediction_metric_label_indices_reject_fractional_near_integer_values() -> None:
    assert _numeric_label_indices(pd.Series([0.0, 1.000001, 2.0])) is None


def test_prediction_metric_label_indices_reject_nonfinite_values() -> None:
    assert _numeric_label_indices(pd.Series([0.0, np.inf, 2.0])) is None
