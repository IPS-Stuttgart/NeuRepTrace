"""Runtime patch for invalid true-label handling in observation ensembles."""

from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd


def _missing_true_labels(output: pd.DataFrame, observation_ensemble: Any) -> list[int]:
    prob_columns = observation_ensemble.probability_columns(output)
    if "true_label" not in output.columns or "probability_true_class" not in output.columns or not prob_columns:
        return []

    true_probability = pd.to_numeric(output["probability_true_class"], errors="coerce")
    if not bool(true_probability.isna().any()):
        return []

    label_values = observation_ensemble._label_values(prob_columns)
    valid_labels = {int(label) for label in label_values}
    true_labels = observation_ensemble._integer_label_values(output["true_label"])
    return sorted({int(label) for label in true_labels if int(label) not in valid_labels})


def install() -> None:
    """Reject ensemble true labels that are not represented by probability columns."""

    import neureptrace.observation_ensemble as observation_ensemble

    if getattr(observation_ensemble.ensemble_probability_observations, "_missing_label_patch", False):
        return

    original_ensemble_probability_observations = observation_ensemble.ensemble_probability_observations

    @wraps(original_ensemble_probability_observations)
    def ensemble_probability_observations(*args: Any, **kwargs: Any):
        output = original_ensemble_probability_observations(*args, **kwargs)
        missing = _missing_true_labels(output, observation_ensemble)
        if missing:
            prob_columns = observation_ensemble.probability_columns(output)
            label_values = observation_ensemble._label_values(prob_columns)
            raise ValueError(f"true_label values must index probability labels {list(label_values)}; missing labels: {missing[:5]}")
        return output

    ensemble_probability_observations._missing_label_patch = True  # type: ignore[attr-defined]
    observation_ensemble.ensemble_probability_observations = ensemble_probability_observations


__all__ = ["install"]
