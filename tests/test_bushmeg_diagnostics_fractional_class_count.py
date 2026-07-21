from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.bushmeg_diagnostics import infer_balanced_accuracy_chance


def test_chance_inference_does_not_truncate_fractional_class_counts() -> None:
    summary = pd.DataFrame({"n_classes": [2.5]})

    with pytest.raises(ValueError, match="Could not infer chance level"):
        infer_balanced_accuracy_chance(summary)


def test_chance_inference_falls_back_when_class_count_is_fractional() -> None:
    summary = pd.DataFrame(
        {
            "n_classes": [2.5],
            "class_names": ["left|right|up|down"],
        }
    )

    assert infer_balanced_accuracy_chance(summary) == pytest.approx(0.25)


@pytest.mark.parametrize("invalid_value", [np.inf, "four"])
def test_chance_inference_falls_back_when_class_count_metadata_is_partly_invalid(invalid_value: object) -> None:
    summary = pd.DataFrame(
        {
            "n_classes": [4, invalid_value],
            "class_names": ["left|right|up", "left|right|up"],
        }
    )

    assert infer_balanced_accuracy_chance(summary) == pytest.approx(1.0 / 3.0)


def test_chance_inference_preserves_missing_and_uniform_integral_class_counts() -> None:
    summary = pd.DataFrame({"n_classes": ["4", 4.0, np.nan]})

    assert infer_balanced_accuracy_chance(summary) == pytest.approx(0.25)
