"""Handle partial class-name metadata during observation ensembling."""

from __future__ import annotations

import importlib
from functools import wraps

import pandas as pd

_PATCH_MARKER = "_neureptrace_observation_ensemble_partial_class_columns_patch_installed"


def _complete_partial_class_columns(observations: pd.DataFrame, observation_ensemble) -> pd.DataFrame:
    """Fill missing class_* columns when some, but not all, probability classes are named."""

    prob_columns = observation_ensemble.probability_columns(observations)
    if not prob_columns:
        return observations
    class_columns = observation_ensemble._class_columns_for_probabilities(observations, prob_columns)
    if not class_columns or len(class_columns) == len(prob_columns):
        return observations

    completed = observations.copy()
    for prob_column in prob_columns:
        suffix = str(prob_column).removeprefix("prob_class_")
        class_column = f"class_{suffix}"
        if class_column not in completed.columns:
            completed[class_column] = suffix
    return completed


def install() -> None:
    """Patch ensemble construction to tolerate partially specified class labels."""

    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    original_ensemble = observation_ensemble.ensemble_probability_observations
    if getattr(original_ensemble, _PATCH_MARKER, False):
        return

    @wraps(original_ensemble)
    def ensemble_probability_observations(observations: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        completed_observations = _complete_partial_class_columns(observations, observation_ensemble)
        return original_ensemble(completed_observations, *args, **kwargs)

    setattr(ensemble_probability_observations, _PATCH_MARKER, True)
    observation_ensemble.ensemble_probability_observations = ensemble_probability_observations


__all__ = ["install"]
