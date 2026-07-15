"""Preserve exact and missing matched-filter group and stream identifiers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from neureptrace._object_label_utils import label_equal_mask, values_equal

_PATCH_MARKER = "_neureptrace_matched_filter_group_keys_patch_installed"


def _is_missing(value: object) -> bool:
    """Return whether *value* is a scalar missing-value sentinel."""

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _identifier_values_equal(left: object, right: object) -> bool:
    """Compare identifiers exactly while treating scalar missing sentinels alike."""

    left_missing = _is_missing(left)
    right_missing = _is_missing(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    return values_equal(left, right)


def _grouped(frame: pd.DataFrame, columns: Sequence[str], *, sort: bool = True):
    """Group rows without discarding missing-valued identifiers."""

    if not columns:
        return [((), frame)]
    by: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    return frame.groupby(by, sort=sort, dropna=False)


def _key_values(key: object, columns: Sequence[str]) -> dict[str, object]:
    """Map group keys without unpacking tuple-valued single-column identifiers."""

    values = (key,) if len(columns) == 1 else key if isinstance(key, tuple) else (key,)
    return dict(zip(columns, values, strict=True))


def _filter_by_values(frame: pd.DataFrame, values: dict[str, object]) -> pd.DataFrame:
    """Filter identifiers using exact equality and missing-sentinel equivalence."""

    filtered = frame
    for column, value in values.items():
        if column not in filtered.columns:
            continue
        if _is_missing(value):
            mask = filtered[column].isna().to_numpy(dtype=bool)
        else:
            mask = label_equal_mask(filtered[column].to_numpy(dtype=object), value)
        filtered = filtered.loc[mask]
    return filtered


def fit_stimulus_event_templates(
    observations: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    template_window: tuple[float, float] = (0.0, 0.35),
    template_step: float | None = None,
    score_mode: str = "class_probability",
    target_classes: Sequence[str | int] | None = None,
    group_columns: Sequence[str] | None = None,
    stream_columns: Sequence[str] | None = None,
    min_template_events: int = 1,
    min_template_coverage: float = 0.8,
) -> pd.DataFrame:
    """Fit templates without stringifying annotation stream identifiers."""

    from neureptrace import matched_filter_detection as matched_filter

    if "time" not in observations.columns:
        raise ValueError("Observation rows must contain a time column.")
    if "onset_time" not in annotations.columns:
        raise ValueError("Template annotations must contain onset_time.")
    if min_template_events < 1:
        raise ValueError("min_template_events must be at least 1.")
    if not 0 < min_template_coverage <= 1:
        raise ValueError("min_template_coverage must be in (0, 1].")

    groups = matched_filter._group_columns(observations, group_columns)
    streams = matched_filter._stream_columns(observations, stream_columns)
    offsets = matched_filter._template_offsets(
        template_window,
        matched_filter._infer_template_step(observations, streams) if template_step is None else float(template_step),
    )
    classes = matched_filter._target_class_table(observations, target_classes)
    rows: list[dict[str, object]] = []
    for group_key, group_frame in _grouped(observations, groups, sort=True):
        group_values = _key_values(group_key, groups)
        group_annotations = _filter_by_values(annotations, group_values)
        if group_annotations.empty:
            continue
        stream_groups = {stream_key: stream_frame for stream_key, stream_frame in _grouped(group_frame, streams, sort=False)}
        for _, class_row in classes.iterrows():
            class_annotations = group_annotations.loc[
                matched_filter._annotation_class_mask(
                    group_annotations,
                    stimulus_label=class_row["stimulus_label"],
                    stimulus_class=str(class_row["stimulus_class"]),
                )
            ]
            if class_annotations.empty:
                continue
            scores = matched_filter._score_values(
                group_frame,
                stimulus_label=class_row["stimulus_label"],
                stimulus_class=str(class_row["stimulus_class"]),
                score_column=str(class_row["score_column"]),
                score_mode=score_mode,
            )
            baseline = float(pd.to_numeric(scores, errors="coerce").dropna().median())
            event_vectors = []
            for _, annotation in class_annotations.iterrows():
                annotation_stream_values = {
                    column: annotation[column]
                    for column in streams
                    if column in annotation.index and not _is_missing(annotation[column])
                }
                matching_streams = stream_groups.items()
                if annotation_stream_values:
                    matching_streams = [
                        (stream_key, stream_frame)
                        for stream_key, stream_frame in stream_groups.items()
                        if all(
                            _identifier_values_equal(_key_values(stream_key, streams)[column], value)
                            for column, value in annotation_stream_values.items()
                        )
                    ]
                for _, stream_frame in matching_streams:
                    table = matched_filter._time_score_table(stream_frame, scores.loc[stream_frame.index])
                    sampled = matched_filter._interpolate(table, float(annotation["onset_time"]) + offsets)
                    if np.isfinite(sampled).mean() >= min_template_coverage:
                        event_vectors.append(sampled)
            if len(event_vectors) < min_template_events:
                continue
            template_values = np.nanmean(np.vstack(event_vectors), axis=0)
            excess = np.where(np.isfinite(template_values), template_values - baseline, 0.0)
            norm = float(np.sqrt(np.sum(excess**2)))
            if not np.isfinite(norm) or norm <= 0:
                continue
            for time, value, weight in zip(offsets, template_values, excess / norm, strict=True):
                rows.append(
                    {
                        **group_values,
                        "stimulus_label": class_row["stimulus_label"],
                        "stimulus_class": class_row["stimulus_class"],
                        "score_column": class_row["score_column"],
                        "score_mode": score_mode,
                        "template_time": float(time),
                        "template_value": float(value) if np.isfinite(value) else np.nan,
                        "template_weight": float(weight),
                        "baseline_score": baseline,
                        "n_template_events": len(event_vectors),
                    }
                )
    return pd.DataFrame(rows)


def _sync_public_alias() -> None:
    public_module = sys.modules.get("neureptrace.stimulus_detection")
    if public_module is not None:
        public_module.fit_stimulus_event_templates = fit_stimulus_event_templates


def install() -> None:
    """Install robust group-key and annotation-stream handling."""

    from neureptrace import matched_filter_detection

    if not getattr(matched_filter_detection, _PATCH_MARKER, False):
        matched_filter_detection._grouped = _grouped  # noqa: SLF001
        matched_filter_detection._key_values = _key_values  # noqa: SLF001
        matched_filter_detection._filter_by_values = _filter_by_values  # noqa: SLF001
        matched_filter_detection.fit_stimulus_event_templates = fit_stimulus_event_templates
        setattr(matched_filter_detection, _PATCH_MARKER, True)
    _sync_public_alias()


__all__ = ["install"]
