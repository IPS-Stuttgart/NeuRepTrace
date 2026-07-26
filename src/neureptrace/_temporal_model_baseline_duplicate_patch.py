"""Reject malformed duplicate time rows in temporal-model baseline windows."""

from __future__ import annotations

import importlib
import inspect
from functools import wraps
from pathlib import Path

import pandas as pd

_PATCH_MARKER = "_neureptrace_temporal_model_baseline_duplicate_patch_installed"


def _validate_baseline_sequence_times(
    temporal_model,
    observations: pd.DataFrame,
    baseline_window: tuple[float, float] | None,
) -> None:
    """Validate baseline sequence identities before optional-baseline fallback."""

    group_columns = temporal_model._model_group_columns(observations)
    for _, decoder_frame in observations.groupby(group_columns, sort=True, dropna=False):
        baseline_frame = temporal_model._filter_time_window(decoder_frame, baseline_window)
        if baseline_frame.empty:
            continue
        key_columns = temporal_model.sequence_key_columns(baseline_frame)
        temporal_model.validate_unique_sequence_times(baseline_frame, key_columns)


def install() -> None:
    """Keep optional empty baselines while surfacing malformed duplicate rows."""

    temporal_model = importlib.import_module("neureptrace.temporal_model")
    if getattr(temporal_model, _PATCH_MARKER, False):
        return

    original_fit_temporal_models = temporal_model.fit_temporal_models
    preflight_reader = inspect.unwrap(temporal_model.read_probability_observations)

    @wraps(original_fit_temporal_models)
    def fit_temporal_models(
        observation_csvs: list[Path],
        *,
        effect_window: tuple[float, float] = (0.1, 0.8),
        baseline_window: tuple[float, float] | None = (-0.1, 0.0),
        n_permutations: int = 100,
        random_seed: int = 13,
        stay_grid_size: int = 200,
        out_summary: Path | None = None,
        out_states: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        effect_window = temporal_model._validate_time_window(effect_window, name="effect_window")
        baseline_window = temporal_model._validate_time_window(baseline_window, name="baseline_window")
        n_permutations = temporal_model._validate_non_negative_integer(n_permutations, name="n_permutations")
        random_seed = temporal_model._validate_non_negative_integer(random_seed, name="random_seed")
        stay_grid_size = temporal_model._validate_integer(stay_grid_size, name="stay_grid_size", minimum=2)
        out_summary, out_states = temporal_model._output_paths(out_summary, out_states)

        paths = [path if isinstance(path, Path) else Path(path) for path in observation_csvs]
        observations = preflight_reader(paths)
        _validate_baseline_sequence_times(temporal_model, observations, baseline_window)

        return original_fit_temporal_models(
            paths,
            effect_window=effect_window,
            baseline_window=baseline_window,
            n_permutations=n_permutations,
            random_seed=random_seed,
            stay_grid_size=stay_grid_size,
            out_summary=out_summary,
            out_states=out_states,
        )

    temporal_model.fit_temporal_models = fit_temporal_models
    setattr(temporal_model, _PATCH_MARKER, True)


__all__ = ["install"]
