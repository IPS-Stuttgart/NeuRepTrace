from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.observation_schema import probability_columns, validate_probability_observations


def test_probability_columns_ignore_non_string_metadata_labels() -> None:
    frame = pd.DataFrame(
        [[7, "run-01", 0.0, 0.25, 0.75]],
        columns=[0, ("metadata", "run"), "time", "prob_class_0", "prob_class_1"],
    )

    assert probability_columns(frame) == ["prob_class_0", "prob_class_1"]

    report = validate_probability_observations(frame)

    assert report.is_valid
    assert report.probability_columns == ("prob_class_0", "prob_class_1")


def test_probability_columns_with_only_non_string_labels_raise_schema_error() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=[0, ("metadata", "value")])

    with pytest.raises(ValueError, match="probability columns"):
        probability_columns(frame)
