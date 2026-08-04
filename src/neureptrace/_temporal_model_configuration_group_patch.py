"""Keep distinct decoder configurations separate in temporal-model fitting."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path

import pandas as pd

_PATCH_MARKER = "_neureptrace_temporal_model_configuration_group_patch_installed"
_MODEL_CONFIGURATION_COLUMNS = (
    "decoder",
    "emission_mode",
    "feature_preprocessor",
    "pca_components",
    "tuned_hyperparameters",
    "tuning_cv_splits",
    "tuning_scoring",
    "tuning_c_grid",
    "temporal_mode",
    "temporal_train_window_start",
    "temporal_train_window_stop",
)


def _iter_groups(frame: pd.DataFrame, columns: list[str]):
    if columns:
        yield from frame.groupby(columns, sort=True, dropna=False)
    else:
        yield (), frame


def _group_values(columns: list[str], key: object) -> dict[str, object]:
    if not columns:
        return {}
    key_values = key if isinstance(key, tuple) else (key,)
    return dict(zip(columns, key_values, strict=True))


def install() -> None:
    """Install configuration-aware temporal fitting and state provenance."""

    temporal_model = importlib.import_module("neureptrace.temporal_model")
    if getattr(temporal_model, _PATCH_MARKER, False):
        return

    temporal_model.MODEL_GROUP_COLUMNS = _MODEL_CONFIGURATION_COLUMNS
    original_build_state_trace = temporal_model.build_state_trace

    @wraps(original_build_state_trace)
    def build_state_trace(
        frame: pd.DataFrame,
        *,
        stay_probability: float,
        class_names: list[str],
        prob_columns: list[str],
    ) -> pd.DataFrame:
        """Decode each model configuration independently and retain its identity."""

        group_columns = temporal_model._model_group_columns(frame)
        state_frames: list[pd.DataFrame] = []
        for key, group in _iter_groups(frame, group_columns):
            states = original_build_state_trace(
                group,
                stay_probability=stay_probability,
                class_names=class_names,
                prob_columns=prob_columns,
            )
            for column, value in _group_values(group_columns, key).items():
                states[column] = value
            state_frames.append(states)
        return pd.concat(state_frames, ignore_index=True) if state_frames else pd.DataFrame()

    @wraps(temporal_model.fit_temporal_models)
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
        """Fit one sticky switching model per complete decoder configuration."""

        effect_window = temporal_model._validate_time_window(effect_window, name="effect_window")
        baseline_window = temporal_model._validate_time_window(baseline_window, name="baseline_window")
        n_permutations = temporal_model._validate_non_negative_integer(n_permutations, name="n_permutations")
        random_seed = temporal_model._validate_non_negative_integer(random_seed, name="random_seed")
        stay_grid_size = temporal_model._validate_integer(stay_grid_size, name="stay_grid_size", minimum=2)
        out_summary, out_states = temporal_model._output_paths(out_summary, out_states)

        observations = temporal_model.read_probability_observations(observation_csvs)
        prob_columns = temporal_model.probability_columns(observations)
        group_columns = temporal_model._model_group_columns(observations)
        rows: list[dict[str, object]] = []
        state_frames: list[pd.DataFrame] = []

        for keys, decoder_frame in _iter_groups(observations, group_columns):
            group_key_values = _group_values(group_columns, keys)
            group_values = {column: str(value) for column, value in group_key_values.items()}
            class_names = temporal_model._class_names(decoder_frame, prob_columns)
            effect_frame = temporal_model._filter_time_window(decoder_frame, effect_window)
            effect_sequences = temporal_model._sequences_from_frame(effect_frame, prob_columns)
            observed_fit = temporal_model.fit_sticky_switching_model(effect_sequences, stay_grid_size=stay_grid_size)
            rows.append(temporal_model._model_row(group_values, "observed_effect", observed_fit))

            baseline_frame = temporal_model._filter_time_window(decoder_frame, baseline_window)
            if not baseline_frame.empty:
                baseline_key_columns = temporal_model.sequence_key_columns(baseline_frame)
                temporal_model.validate_unique_sequence_times(baseline_frame, baseline_key_columns)
                try:
                    baseline_sequences = temporal_model._sequences_from_frame(baseline_frame, prob_columns)
                except ValueError:
                    baseline_sequences = []
                if baseline_sequences:
                    baseline_fit = temporal_model.fit_sticky_switching_model(baseline_sequences, stay_grid_size=stay_grid_size)
                    rows.append(temporal_model._model_row(group_values, "baseline_window", baseline_fit))

            if n_permutations > 0:
                for offset, control in enumerate(("shuffled_time", "shuffled_label")):
                    control_fits = temporal_model._fit_control(
                        effect_sequences,
                        control=control,
                        n_permutations=n_permutations,
                        random_seed=random_seed + offset,
                        stay_grid_size=stay_grid_size,
                    )
                    rows.append(
                        temporal_model._control_row(
                            group_values,
                            control,
                            control_fits,
                            observed_gain=observed_fit["persistence_gain_per_observation"],
                        )
                    )

            if out_states is not None:
                state_frames.append(
                    build_state_trace(
                        effect_frame,
                        stay_probability=observed_fit["best_stay_probability"],
                        class_names=class_names,
                        prob_columns=prob_columns,
                    )
                )

        summary = pd.DataFrame(rows)
        if out_summary is not None:
            out_summary.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out_summary, index=False)

        states = pd.concat(state_frames, ignore_index=True) if state_frames else None
        if out_states is not None and states is not None:
            out_states.parent.mkdir(parents=True, exist_ok=True)
            states.to_csv(out_states, index=False)
        return summary, states

    temporal_model.build_state_trace = build_state_trace
    temporal_model.fit_temporal_models = fit_temporal_models
    setattr(temporal_model, _PATCH_MARKER, True)


__all__ = ["install"]
