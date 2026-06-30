from __future__ import annotations

import pytest

from neureptrace.decoding import source_temperature


def test_fit_source_temperature_scaling_reports_missing_list_label() -> None:
    with pytest.raises(ValueError, match="absent from classes"):
        source_temperature.fit_source_temperature_scaling(
            source_probabilities=[[0.9, 0.1], [0.1, 0.9]],
            source_labels=[[1, 1], [9, 9]],
            test_probabilities=[[0.5, 0.5]],
            classes=[[1, 1], [2, 2]],
        )
