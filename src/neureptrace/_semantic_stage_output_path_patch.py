"""Apply semantic-stage compatibility fixes."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_OUTPUT_PATCH_MARKER = "_neureptrace_semantic_stage_output_path_patch_installed"
_SEQUENCE_KEY_PATCH_MARKER = "_neureptrace_semantic_stage_sequence_key_patch_installed"
_STATE_NAME_PATCH_MARKER = "_neureptrace_semantic_stage_state_name_patch_installed"
_TEMPORAL_STATE_PEAK_PATCH_MARKER = "_neureptrace_temporal_state_peak_selection_patch_installed"
_SUBJECT_FALLBACK_PATCH_MARKER = "_neureptrace_semantic_stage_subject_fallback_patch_installed"
_NUMERIC_COLUMN_PATCH_MARKER = "_neureptrace_semantic_stage_numeric_column_patch_installed"
_MISSING_SEQUENCE_COMPONENT = object()


def _output_paths(
    out_time: Path | str | None,
    out_stages: Path | str | None,
    out_report: Path | str | None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Normalize semantic-stage outputs and reject colliding destinations."""

    paths = (
        None if out_time is None else Path(out_time),
        None if out_stages is None else Path(out_stages),
        None if out_report is None else Path(out_report),
    )
    labels = ("time summary", "stage intervals", "report")
    destinations: dict[Path, str] = {}
    for label, path in zip(labels, paths, strict=True):
        if path is None:
            continue
        destination = path.resolve(strict=False)
        previous = destinations.get(destination)
        if previous is not None:
            raise ValueError(
                "Semantic-stage output paths must be distinct; "
                f"{previous} and {label} both resolve to {destination}."
            )
        destinations[destination] = label
    return paths


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _is_boolean_or_complex_scalar(value: object) -> bool:
    """Return whether numeric coercion would alter an unsupported scalar."""

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _is_boolean_or_complex_scalar(value.item())
    return isinstance(value, (bool, np.bool_, complex, np.complexfloating))


def _state_trace_subject_fallbacks(
    csv_paths: list[Path | str],
) -> tuple[list[Path], np.ndarray, list[str]]:
    """Return input paths and row-aligned subject fallback metadata."""

    paths = [path if isinstance(path, Path) else Path(path) for path in csv_paths]
    missing_subjects: list[bool] = []
    fallback_subjects: list[str] = []
    for csv_path in paths:
        frame = pd.read_csv(csv_path)
        fallback_subjects.extend([csv_path.stem] * len(frame))
        if "subject" not in frame.columns:
            missing_subjects.extend([False] * len(frame))
        else:
            missing_subjects.extend(frame["subject"].isna().tolist())
    return paths, np.asarray(missing_subjects, dtype=bool), fallback_subjects


def _sequence_keys(frame: pd.DataFrame) -> pd.Series:
    """Return collision-safe, type-preserving semantic-stage sequence keys."""

    semantic_stages = importlib.import_module("neureptrace.semantic_stages")
    key_columns = [
        *semantic_stages._stage_group_columns(frame),
        *(
            column
            for column in semantic_stages.sequence_key_columns(frame)
            if column not in semantic_stages.STAGE_GROUP_COLUMNS
        ),
    ]
    semantic_stages.validate_unique_sequence_times(frame, key_columns)
    keys = [
        tuple(
            _MISSING_SEQUENCE_COMPONENT if _is_missing_scalar(value) else value
            for value in values
        )
        for values in frame[key_columns].itertuples(index=False, name=None)
    ]
    return pd.Series(keys, index=frame.index, dtype=object)


def _validate_state_label_columns(frame: pd.DataFrame, posterior_columns: list[str]) -> None:
    """Reject state columns whose class meaning changes across trace rows."""

    for posterior_column in posterior_columns:
        suffix = posterior_column.removeprefix("posterior_state_")
        state_column = f"state_{suffix}"
        if state_column not in frame.columns:
            continue
        labels: list[str] = []
        for value in frame[state_column]:
            if _is_missing_scalar(value):
                continue
            label = str(value)
            if label not in labels:
                labels.append(label)
        if len(labels) > 1:
            raise ValueError(
                f"{state_column} values must identify one state for {posterior_column}; "
                f"found {labels}."
            )


def _validate_unique_state_names(state_names: list[str]) -> None:
    """Reject duplicate labels that make posterior-state columns ambiguous."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for state_name in state_names:
        if state_name in seen and state_name not in duplicates:
            duplicates.append(state_name)
        seen.add(state_name)
    if duplicates:
        raise ValueError(
            "State labels must map uniquely to posterior columns; "
            f"duplicate labels: {duplicates}."
        )


def _finite_posterior_rows(stage_time: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot participate in temporal-state peak selection."""

    column = "posterior_true_class_mean"
    if stage_time.empty or column not in stage_time.columns:
        return stage_time
    numeric = pd.to_numeric(stage_time[column], errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if bool(finite.all()):
        return stage_time
    filtered = stage_time.loc[finite].copy()
    filtered[column] = numeric.loc[finite].to_numpy(dtype=float)
    return filtered


def _install_temporal_state_peak_selection() -> None:
    """Make temporal-state peak metadata independent of invalid index labels and values."""

    workflow = importlib.import_module("neureptrace.temporal_state_workflow")
    original_stage_stats = workflow._stage_stats
    if getattr(original_stage_stats, _TEMPORAL_STATE_PEAK_PATCH_MARKER, False):
        return

    @wraps(original_stage_stats)
    def stage_stats(
        stages: pd.DataFrame,
        stage_time: pd.DataFrame,
        task: str,
        decoder: str,
        emission_mode: str,
    ) -> dict[str, float | int]:
        stage_time = _finite_posterior_rows(stage_time)
        if not stage_time.index.is_unique:
            stage_time = stage_time.reset_index(drop=True)
        return original_stage_stats(stages, stage_time, task, decoder, emission_mode)

    setattr(stage_stats, _TEMPORAL_STATE_PEAK_PATCH_MARKER, True)
    workflow._stage_stats = stage_stats


def install() -> None:
    """Patch semantic-stage identities, state labels, output destinations, and summaries."""

    semantic_stages = importlib.import_module("neureptrace.semantic_stages")

    original_coerce_finite_numeric_column = semantic_stages._coerce_finite_numeric_column
    if not getattr(original_coerce_finite_numeric_column, _NUMERIC_COLUMN_PATCH_MARKER, False):

        @wraps(original_coerce_finite_numeric_column)
        def coerce_finite_numeric_column(
            frame: pd.DataFrame,
            column: str,
            *,
            source: Path | None = None,
        ) -> None:
            values = frame[column]
            if isinstance(values, pd.Series) and bool(values.map(_is_boolean_or_complex_scalar).any()):
                prefix = f"{source} " if source is not None else ""
                raise ValueError(f"{prefix}{column} values must be numeric.")
            original_coerce_finite_numeric_column(frame, column, source=source)

        setattr(coerce_finite_numeric_column, _NUMERIC_COLUMN_PATCH_MARKER, True)
        semantic_stages._coerce_finite_numeric_column = coerce_finite_numeric_column

    original_read_state_traces = semantic_stages.read_state_traces
    if not getattr(original_read_state_traces, _SUBJECT_FALLBACK_PATCH_MARKER, False):

        @wraps(original_read_state_traces)
        def read_state_traces(csv_paths: list[Path | str]) -> pd.DataFrame:
            paths, missing_subjects, fallback_subjects = _state_trace_subject_fallbacks(csv_paths)
            traces = original_read_state_traces(paths)
            if len(traces) != len(missing_subjects):
                raise RuntimeError("Semantic-stage trace row count changed while restoring subject identifiers.")
            if not bool(missing_subjects.any()):
                return traces
            restored = traces.copy()
            fallback_values = pd.Series(fallback_subjects, index=restored.index, dtype=object)
            restored.loc[missing_subjects, "subject"] = fallback_values.loc[missing_subjects]
            restored["subject"] = restored["subject"].astype(str)
            return restored

        setattr(read_state_traces, _SUBJECT_FALLBACK_PATCH_MARKER, True)
        semantic_stages.read_state_traces = read_state_traces

    current_sequence_keys = semantic_stages._sequence_keys
    if not getattr(current_sequence_keys, _SEQUENCE_KEY_PATCH_MARKER, False):
        setattr(_sequence_keys, _SEQUENCE_KEY_PATCH_MARKER, True)
        semantic_stages._sequence_keys = _sequence_keys

    original_state_names = semantic_stages._state_names
    if not getattr(original_state_names, _STATE_NAME_PATCH_MARKER, False):

        @wraps(original_state_names)
        def state_names(frame: pd.DataFrame, columns: list[str]) -> list[str]:
            _validate_state_label_columns(frame, columns)
            names = original_state_names(frame, columns)
            _validate_unique_state_names(names)
            return names

        setattr(state_names, _STATE_NAME_PATCH_MARKER, True)
        semantic_stages._state_names = state_names

    _install_temporal_state_peak_selection()

    original_analyze = semantic_stages.analyze_semantic_stages
    if getattr(original_analyze, _OUTPUT_PATCH_MARKER, False):
        return

    @wraps(original_analyze)
    def analyze_semantic_stages(
        state_trace_csvs: list[Path],
        *,
        posterior_threshold: float = 0.6,
        match_threshold: float = 0.6,
        min_duration: float = 0.04,
        out_time: Path | str | None = None,
        out_stages: Path | str | None = None,
        out_report: Path | str | None = None,
    ) -> tuple[Any, Any, str | None]:
        out_time, out_stages, out_report = _output_paths(out_time, out_stages, out_report)
        return original_analyze(
            state_trace_csvs,
            posterior_threshold=posterior_threshold,
            match_threshold=match_threshold,
            min_duration=min_duration,
            out_time=out_time,
            out_stages=out_stages,
            out_report=out_report,
        )

    setattr(analyze_semantic_stages, _OUTPUT_PATCH_MARKER, True)
    semantic_stages.analyze_semantic_stages = analyze_semantic_stages


__all__ = ["install"]
