"""Handle partial class-name metadata during observation ensembling."""

from __future__ import annotations

import importlib
from functools import wraps

import pandas as pd

_PATCH_MARKER = "_neureptrace_observation_ensemble_partial_class_columns_patch_installed"


def _class_fill_value(values: pd.Series, suffix: str) -> str:
    """Return a stable replacement for missing values in one class-name column."""

    known_values = values.dropna().map(str).drop_duplicates()
    if len(known_values) == 1:
        return str(known_values.iloc[0])
    return suffix


def _complete_partial_class_columns(observations: pd.DataFrame, observation_ensemble) -> pd.DataFrame:
    """Fill missing class_* columns and row values when probability classes are partly named."""

    prob_columns = observation_ensemble.probability_columns(observations)
    if not prob_columns:
        return observations
    class_columns = observation_ensemble._class_columns_for_probabilities(observations, prob_columns)
    if not class_columns:
        return observations

    completed = observations.copy()
    changed = False
    for prob_column in prob_columns:
        suffix = str(prob_column).removeprefix("prob_class_")
        class_column = f"class_{suffix}"
        if class_column not in completed.columns:
            completed[class_column] = suffix
            changed = True
            continue

        missing_mask = completed[class_column].isna()
        if bool(missing_mask.any()):
            completed[class_column] = completed[class_column].astype(object)
            completed.loc[missing_mask, class_column] = _class_fill_value(completed[class_column], suffix)
            changed = True
    return completed if changed else observations


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
