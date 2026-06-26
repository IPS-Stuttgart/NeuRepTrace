from __future__ import annotations

import pytest

import neureptrace  # noqa: F401 - importing installs runtime guardrail patches
from neureptrace.decoding.conditional_coral import conditional_coral_config, fit_pseudo_label_conditional_coral


def test_conditional_coral_config_parses_center_string_values() -> None:
    assert conditional_coral_config(center="false").center is False
    assert conditional_coral_config(center="off").center is False
    assert conditional_coral_config(center="0").center is False
    assert conditional_coral_config(center="true").center is True
    assert conditional_coral_config(center=1).center is True
    assert conditional_coral_config(center=0).center is False


def test_conditional_coral_config_rejects_ambiguous_center_string() -> None:
    with pytest.raises(ValueError, match="center must be a boolean value"):
        conditional_coral_config(center="definitely")


def test_conditional_coral_mapping_config_can_disable_recentering() -> None:
    result = fit_pseudo_label_conditional_coral(
        source_features=[[0.0], [1.0], [4.0], [5.0]],
        source_labels=["a", "a", "b", "b"],
        target_features=[[10.0], [11.0], [14.0], [15.0]],
        target_pseudo_labels=["a", "a", "b", "b"],
        config={"center": "false", "min_target_rows_per_class": 1},
    )

    assert result.metadata["conditional_coral_center"] is False
