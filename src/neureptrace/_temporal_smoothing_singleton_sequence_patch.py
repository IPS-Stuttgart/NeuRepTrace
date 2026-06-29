"""Preserve singleton sequences and reject malformed temporal inputs."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_temporal_smoothing_singleton_sequence_patch_installed"


def _contains_boolean_values(values: object) -> bool:
    """Return whether an array-like object contains Python/NumPy booleans."""

    array = np.asarray(values, dtype=object)
    if array.size == 0:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in array.ravel())


def _reject_boolean_values(values: object, message: str) -> None:
    if _contains_boolean_values(values):
        raise ValueError(message)


def _temporal_config_scalar(value: object, message: str) -> object:
    """Return a scalar config value while rejecting boolean and array-valued inputs."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    elif isinstance(value, (list, tuple, dict, set)):
        raise ValueError(message)
    return value


def install() -> None:
    """Patch temporal smoothing so valid one-row sequences are retained."""

    temporal_model = importlib.import_module("neureptrace.temporal_model")
    temporal_smoothing = importlib.import_module("neureptrace.temporal_smoothing")
    original_smooth = temporal_smoothing.smooth_probability_observations
    if getattr(original_smooth, _PATCH_MARKER, False):
        return

    original_validate_integer = temporal_model._validate_integer

    @wraps(original_validate_integer)
    def _validate_integer(value: object, *, name: str, minimum: int | None = None) -> int:
        value = _temporal_config_scalar(value, f"{name} must be an integer.")
        return original_validate_integer(value, name=name, minimum=minimum)

    temporal_model._validate_integer = _validate_integer

    original_validate_finite_float = temporal_model._validate_finite_float

    @wraps(original_validate_finite_float)
    def _validate_finite_float(value: object, *, name: str) -> float:
        message = f"{name} must be finite."
        value = _temporal_config_scalar(value, message)
        try:
            return original_validate_finite_float(value, name=name)
        except TypeError as exc:
            raise ValueError(message) from exc

    temporal_model._validate_finite_float = _validate_finite_float

    original_validate_probability_matrix = temporal_model._validate_probability_matrix

    @wraps(original_validate_probability_matrix)
    def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
        _reject_boolean_values(
            probabilities,
            "Probability observations must be numeric probabilities, not boolean values.",
        )
        return original_validate_probability_matrix(probabilities)

    temporal_model._validate_probability_matrix = _validate_probability_matrix

    original_read_probability_observations = temporal_model.read_probability_observations

    @wraps(original_read_probability_observations)
    def read_probability_observations(csv_paths: list[Path]) -> pd.DataFrame:
        paths = [path if isinstance(path, Path) else Path(path) for path in csv_paths]
        for csv_path in paths:
            frame = pd.read_csv(csv_path)
            if "time" in frame.columns:
                _reject_boolean_values(
                    frame["time"].to_numpy(dtype=object),
                    f"{csv_path} time values must be numeric, not boolean.",
                )
        return original_read_probability_observations(paths)

    temporal_model.read_probability_observations = read_probability_observations
    temporal_smoothing.read_probability_observations = read_probability_observations

    original_numeric_label_values = temporal_smoothing._numeric_label_values

    @wraps(original_numeric_label_values)
    def _numeric_label_values(frame: pd.DataFrame, label_values: tuple[int, ...]) -> np.ndarray:
        if "true_label" in frame.columns:
            _reject_boolean_values(
                frame["true_label"].to_numpy(dtype=object),
                "true_label values must be numeric integer labels, not boolean values.",
            )
        return original_numeric_label_values(frame, label_values)

    temporal_smoothing._numeric_label_values = _numeric_label_values

    original_metrics_from_probability_observations = temporal_smoothing.metrics_from_probability_observations

    @wraps(original_metrics_from_probability_observations)
    def metrics_from_probability_observations(observations: pd.DataFrame, *, ece_bins: int = 10) -> pd.DataFrame:
        if "true_label" in observations.columns:
            _reject_boolean_values(
                observations["true_label"].to_numpy(dtype=object),
                "true_label values must be numeric integer labels, not boolean values.",
            )
        for column in temporal_smoothing.probability_columns(observations):
            _reject_boolean_values(
                observations[column].to_numpy(dtype=object),
                f"{column} values must be numeric probabilities, not boolean values.",
            )
        return original_metrics_from_probability_observations(observations, ece_bins=ece_bins)

    temporal_smoothing.metrics_from_probability_observations = metrics_from_probability_observations

    @wraps(original_smooth)
    def smooth_probability_observations(
        observation_csvs: list[Path],
        *,
        fit_window: tuple[float, float] | None = temporal_smoothing.DEFAULT_FIT_WINDOW,
        stay_grid_size: int = 200,
        mode: str = "forward_backward",
        apply_window: tuple[float, float] | None = None,
        emission_suffix: str = temporal_smoothing.DEFAULT_EMISSION_SUFFIX,
        ece_bins: int = 10,
        out_observations: Path | None = None,
        out_metrics: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Replace decoder probabilities by sticky-switching temporal posteriors.

        This mirrors ``neureptrace.temporal_smoothing.smooth_probability_observations``
        but keeps valid singleton sequences in the output.  Singleton sequences do
        not contribute temporal-transition evidence, yet dropping them changes the
        evaluation set and can silently bias downstream metrics.
        """

        if mode not in temporal_smoothing.SMOOTHING_MODE_CHOICES:
            choices = ", ".join(temporal_smoothing.SMOOTHING_MODE_CHOICES)
            raise ValueError(f"Unknown temporal smoothing mode '{mode}'. Available modes: {choices}.")
        if mode == "poststimulus_forward_only" and apply_window is None:
            apply_window = temporal_smoothing.DEFAULT_POSTSTIMULUS_APPLY_WINDOW
        elif mode != "poststimulus_forward_only":
            apply_window = None
        smoothing_method = temporal_smoothing.SMOOTHING_METHODS[mode]

        observations = temporal_smoothing.read_probability_observations(observation_csvs).copy()
        observations["__input_order"] = np.arange(len(observations))
        prob_columns = temporal_smoothing.probability_columns(observations)
        group_columns = temporal_smoothing._smoothing_group_columns(observations) or temporal_smoothing._model_group_columns(observations)
        smoothed_frames: list[pd.DataFrame] = []

        for _, decoder_frame in temporal_smoothing._iter_groups(observations, group_columns):
            fit_frame = temporal_smoothing._filter_time_window(decoder_frame, fit_window) if fit_window is not None else decoder_frame.copy()
            fit_sequences = temporal_smoothing._sequences_from_frame(fit_frame, prob_columns)
            fit = temporal_smoothing.fit_sticky_switching_model(fit_sequences, stay_grid_size=stay_grid_size)
            stay_probability = float(fit["best_stay_probability"])
            class_names = temporal_smoothing._class_names(decoder_frame, prob_columns)
            key_columns = temporal_smoothing.sequence_key_columns(decoder_frame)
            temporal_smoothing.validate_unique_sequence_times(decoder_frame, key_columns)

            for _, sequence_frame in decoder_frame.sort_values([*key_columns, "time"]).groupby(key_columns, sort=True, dropna=False):
                probabilities = temporal_smoothing._normalize_probabilities(sequence_frame[prob_columns].to_numpy(dtype=float))
                if len(probabilities) < 2:
                    posterior = probabilities
                else:
                    posterior = temporal_smoothing._smooth_sequence_posteriors(
                        sequence_frame,
                        probabilities,
                        stay_probability=stay_probability,
                        mode=mode,
                        apply_window=apply_window,
                    )
                smoothed_frames.append(
                    temporal_smoothing._with_posterior_columns(
                        sequence_frame,
                        posterior,
                        prob_columns=prob_columns,
                        class_names=class_names,
                        stay_probability=stay_probability,
                        fit_window=fit_window,
                        apply_window=apply_window,
                        emission_suffix=emission_suffix,
                        smoothing_method=smoothing_method,
                    )
                )

        if not smoothed_frames:
            raise ValueError("Need at least one sequence with two or more time points for temporal smoothing.")

        smoothed = pd.concat(smoothed_frames, ignore_index=True).sort_values("__input_order").drop(columns=["__input_order"]).reset_index(drop=True)
        metrics = temporal_smoothing.metrics_from_probability_observations(smoothed, ece_bins=ece_bins)

        if out_observations is not None:
            out_observations.parent.mkdir(parents=True, exist_ok=True)
            smoothed.to_csv(out_observations, index=False)
        if out_metrics is not None:
            out_metrics.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(out_metrics, index=False)
        return smoothed, metrics

    setattr(smooth_probability_observations, _PATCH_MARKER, True)
    temporal_smoothing.smooth_probability_observations = smooth_probability_observations


__all__ = ["install"]
