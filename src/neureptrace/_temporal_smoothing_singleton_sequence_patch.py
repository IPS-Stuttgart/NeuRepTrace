"""Preserve singleton sequences during temporal posterior smoothing."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_temporal_smoothing_singleton_sequence_patch_installed"


def install() -> None:
    """Patch temporal smoothing so valid one-row sequences are retained."""

    temporal_smoothing = importlib.import_module("neureptrace.temporal_smoothing")
    original_smooth = temporal_smoothing.smooth_probability_observations
    if getattr(original_smooth, _PATCH_MARKER, False):
        return

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
