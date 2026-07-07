from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace._onset_utils import is_correct_detection, prediction_value


def test_prediction_value_accepts_composite_predicted_label() -> None:
    row = pd.Series({"predicted_label": ("face", np.nan), "predicted_class": "fallback"})

    assert prediction_value(row) == ("face", np.nan)


def test_prediction_value_falls_back_when_composite_label_is_all_missing() -> None:
    row = pd.Series({"predicted_label": (np.nan, np.nan), "predicted_class": "fallback"})

    assert prediction_value(row) == "fallback"


def test_is_correct_detection_compares_composite_labels_without_ambiguous_truth_value() -> None:
    correct = pd.Series({"true_label": ("face", np.nan), "predicted_label": ("face", np.float64(np.nan))})
    incorrect = pd.Series({"true_label": ("face", 1), "predicted_label": ("scene", 1)})

    assert is_correct_detection(correct) is True
    assert is_correct_detection(incorrect) is False
