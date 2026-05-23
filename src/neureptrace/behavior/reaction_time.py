"""Generic reaction-time loading, joins, and metric associations.

These helpers keep the reusable parts of the deprecated PyMEGDec alpha/RT
workflow without depending on alpha-specific feature extraction.  They operate on
plain row dictionaries so they can be used with NeuRepTrace CSV outputs,
probability-observation summaries, or external behavioral tables.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REACTION_TIME_FIELD_CANDIDATES = (
    "reaction_time",
    "reaction_time_s",
    "response_time",
    "response_time_s",
    "rt",
)
TRIAL_INDEX_BASE_CHOICES = (0, 1)


class ReactionTimeUnavailableError(ValueError):
    """Raised when reaction times are not present in the available metadata."""


@dataclass(frozen=True)
class ReactionTimeCsvConfig:
    """Column mapping for an external reaction-time CSV.

    ``trial_index_base`` describes the CSV's trial numbering. NeuRepTrace row
    tables conventionally use zero-based trial indices; set
    ``trial_index_base=1`` for behavioral CSVs numbered 1..N.
    """

    participant_column: str | None = None
    trial_column: str | None = None
    reaction_time_column: str | None = None
    dataset_column: str | None = None
    default_participant: int | str | None = None
    default_dataset: str = "main"
    reaction_time_scale: float = 1.0
    trial_index_base: int = 0


def _clean_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _to_float(value: object) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value: object) -> int:
    return int(float(str(value).strip()))


def _validate_trial_index_base(trial_index_base: int) -> int:
    if trial_index_base not in TRIAL_INDEX_BASE_CHOICES:
        raise ValueError(f"trial_index_base must be one of {TRIAL_INDEX_BASE_CHOICES}, got {trial_index_base!r}.")
    return trial_index_base


def _normalize_csv_trial(value: object, trial_index_base: int) -> int:
    raw_trial = _to_int(value)
    zero_based_trial = raw_trial - _validate_trial_index_base(trial_index_base)
    if zero_based_trial < 0:
        raise ValueError(
            f"CSV trial value {raw_trial!r} with trial_index_base={trial_index_base} maps to a negative zero-based trial index."
        )
    return zero_based_trial


def _column(fieldnames: Sequence[str], explicit: str | None, candidates: Sequence[str], *, required: bool = True) -> str | None:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"CSV column {explicit!r} was not found.")
        return explicit

    lookup = {field_name.lower(): field_name for field_name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(f"CSV must contain one of these columns: {', '.join(candidates)}.")
    return None


def load_reaction_time_csv(path: str | Path, config: ReactionTimeCsvConfig | None = None) -> list[dict[str, object]]:
    """Load external reaction times and normalize join key columns."""

    config = config or ReactionTimeCsvConfig()
    trial_index_base = _validate_trial_index_base(config.trial_index_base)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        participant_column = _column(
            fieldnames,
            config.participant_column,
            ("participant", "participant_id", "subject", "subject_id", "part"),
            required=config.default_participant is None,
        )
        trial_column = _column(fieldnames, config.trial_column, ("trial", "trial_idx", "trial_index"))
        rt_column = _column(fieldnames, config.reaction_time_column, REACTION_TIME_FIELD_CANDIDATES)
        dataset_column = _column(
            fieldnames,
            config.dataset_column,
            ("dataset", "condition", "source", "split"),
            required=False,
        )

        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "participant": _clean_id(row[participant_column] if participant_column else config.default_participant),
                    "dataset": str(row[dataset_column] if dataset_column else config.default_dataset),
                    "trial": _normalize_csv_trial(row[trial_column], trial_index_base),
                    "reaction_time": _to_float(row[rt_column]) * config.reaction_time_scale,
                }
            )
    return rows


def reaction_time_rows_from_values(
    values: Sequence[float],
    *,
    participant: int | str | None = None,
    dataset: str = "main",
    reaction_time_scale: float = 1.0,
) -> list[dict[str, object]]:
    """Create normalized reaction-time rows from one value per trial."""

    values = np.asarray(values, dtype=float).ravel()
    return [
        {
            "participant": _clean_id(participant),
            "dataset": dataset,
            "trial": trial_idx,
            "reaction_time": float(values[trial_idx]) * reaction_time_scale,
        }
        for trial_idx in range(values.size)
    ]


def extract_reaction_times_from_metadata(
    metadata: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    *,
    reaction_time_column: str | None = None,
    participant: int | str | None = None,
    dataset: str = "main",
    reaction_time_scale: float = 1.0,
) -> list[dict[str, object]]:
    """Extract reaction times from a metadata mapping or row sequence.

    The helper accepts a pandas ``DataFrame`` because data frames expose a
    mapping-like column interface, but pandas is not required.
    """

    if hasattr(metadata, "columns") and hasattr(metadata, "__getitem__"):
        columns = tuple(str(column) for column in metadata.columns)
        rt_column = _column(columns, reaction_time_column, REACTION_TIME_FIELD_CANDIDATES)
        values = metadata[rt_column]  # type: ignore[index]
    elif isinstance(metadata, Mapping):
        columns = tuple(str(column) for column in metadata.keys())
        rt_column = _column(columns, reaction_time_column, REACTION_TIME_FIELD_CANDIDATES)
        values = metadata[rt_column]  # type: ignore[index]
    else:
        rows = list(metadata)
        if not rows:
            raise ReactionTimeUnavailableError("metadata contains no rows.")
        columns = tuple(str(column) for column in rows[0].keys())
        rt_column = _column(columns, reaction_time_column, REACTION_TIME_FIELD_CANDIDATES)
        values = [row[rt_column] for row in rows]
    return reaction_time_rows_from_values(values, participant=participant, dataset=dataset, reaction_time_scale=reaction_time_scale)


def _join_key(row: Mapping[str, object], *, participant_column: str, dataset_column: str, trial_column: str) -> tuple[str, str, int]:
    return (
        _clean_id(row.get(participant_column)),
        str(row.get(dataset_column, "main")),
        _to_int(row.get(trial_column)),
    )


def _trial_group_key(row: Mapping[str, object], *, participant_column: str, dataset_column: str) -> tuple[str, str]:
    return (_clean_id(row.get(participant_column)), str(row.get(dataset_column, "main")))


def _group_trials_by_participant_dataset(
    rows: Iterable[Mapping[str, object]],
    *,
    participant_column: str,
    dataset_column: str,
    trial_column: str,
) -> dict[tuple[str, str], set[int]]:
    grouped: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        grouped.setdefault(_trial_group_key(row, participant_column=participant_column, dataset_column=dataset_column), set()).add(_to_int(row.get(trial_column)))
    return grouped


def _raise_if_likely_one_based_reaction_trials(
    rows: Sequence[Mapping[str, object]],
    reaction_time_rows: Sequence[Mapping[str, object]],
    *,
    participant_column: str,
    dataset_column: str,
    trial_column: str,
) -> None:
    row_trials_by_group = _group_trials_by_participant_dataset(
        rows,
        participant_column=participant_column,
        dataset_column=dataset_column,
        trial_column=trial_column,
    )
    reaction_trials_by_group = _group_trials_by_participant_dataset(
        reaction_time_rows,
        participant_column=participant_column,
        dataset_column=dataset_column,
        trial_column=trial_column,
    )
    for participant, dataset in sorted(row_trials_by_group.keys() & reaction_trials_by_group.keys()):
        row_trials = row_trials_by_group[(participant, dataset)]
        reaction_trials = reaction_trials_by_group[(participant, dataset)]
        if not row_trials or not reaction_trials:
            continue
        current_matches = len(row_trials & reaction_trials)
        one_based_matches = len(row_trials & {trial - 1 for trial in reaction_trials})
        max_possible_matches = min(len(row_trials), len(reaction_trials))
        if one_based_matches > current_matches and one_based_matches >= max_possible_matches:
            raise ValueError(
                "Reaction-time trial numbers for "
                f"participant {participant!r}, dataset {dataset!r} look one-based. "
                "Use ReactionTimeCsvConfig(trial_index_base=1) or convert the trial column before joining."
            )


def join_reaction_times(
    rows: Iterable[Mapping[str, object]],
    reaction_time_rows: Iterable[Mapping[str, object]],
    *,
    participant_column: str = "participant",
    dataset_column: str = "dataset",
    trial_column: str = "trial",
    reaction_time_column: str = "reaction_time",
    require_match: bool = True,
    detect_one_based_trials: bool = True,
) -> list[dict[str, object]]:
    """Join metric rows with reaction times by participant, dataset, and trial."""

    rows = list(rows)
    reaction_time_rows = list(reaction_time_rows)
    if detect_one_based_trials:
        _raise_if_likely_one_based_reaction_trials(
            rows,
            reaction_time_rows,
            participant_column=participant_column,
            dataset_column=dataset_column,
            trial_column=trial_column,
        )

    reaction_by_key: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for row in reaction_time_rows:
        key = _join_key(row, participant_column=participant_column, dataset_column=dataset_column, trial_column=trial_column)
        if key in reaction_by_key:
            raise ValueError(f"Duplicate reaction-time row for key {key}.")
        reaction_by_key[key] = row

    joined_rows: list[dict[str, object]] = []
    for row in rows:
        key = _join_key(row, participant_column=participant_column, dataset_column=dataset_column, trial_column=trial_column)
        reaction_row = reaction_by_key.get(key)
        if reaction_row is None:
            if require_match:
                raise ValueError(f"No reaction-time row for key {key}.")
            continue
        joined_row = dict(row)
        joined_row[reaction_time_column] = reaction_row[reaction_time_column]
        joined_rows.append(joined_row)

    if not joined_rows:
        raise ValueError("No rows matched the reaction-time rows.")
    return joined_rows


def _finite_metric_arrays(rows: Sequence[Mapping[str, object]], metric: str, reaction_time_column: str) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.array([_to_float(row.get(metric)) for row in rows], dtype=float)
    y_values = np.array([_to_float(row.get(reaction_time_column)) for row in rows], dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid], y_values[valid]


def _empty_association(scope: str, participant: str, metric: str, x_values: np.ndarray, y_values: np.ndarray) -> dict[str, object]:
    return {
        "scope": scope,
        "participant": participant,
        "metric": metric,
        "n_trials": int(x_values.size),
        "metric_mean": float(np.mean(x_values)) if x_values.size else np.nan,
        "reaction_time_mean": float(np.mean(y_values)) if y_values.size else np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "slope_reaction_time_per_unit": np.nan,
        "intercept_reaction_time": np.nan,
    }


def _association_row(
    scope: str,
    participant: str,
    metric: str,
    rows: Sequence[Mapping[str, object]],
    *,
    reaction_time_column: str,
    min_trials: int,
) -> dict[str, object]:
    x_values, y_values = _finite_metric_arrays(rows, metric, reaction_time_column)
    result = _empty_association(scope, participant, metric, x_values, y_values)
    if x_values.size < min_trials or np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        return result

    from scipy import stats  # pylint: disable=import-outside-toplevel

    pearson = stats.pearsonr(x_values, y_values)
    regression = stats.linregress(x_values, y_values)
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "slope_reaction_time_per_unit": float(regression.slope),
            "intercept_reaction_time": float(regression.intercept),
        }
    )
    return result


def _group_by_participant(rows: Sequence[Mapping[str, object]], participant_column: str) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_clean_id(row.get(participant_column)), []).append(row)
    return grouped


def _within_participant_centered_rows(
    grouped_rows: Mapping[str, Sequence[Mapping[str, object]]],
    metric: str,
    reaction_time_column: str,
) -> list[dict[str, object]]:
    centered_rows: list[dict[str, object]] = []
    for participant_rows in grouped_rows.values():
        x_values, y_values = _finite_metric_arrays(participant_rows, metric, reaction_time_column)
        for x_value, y_value in zip(x_values, y_values):
            centered_rows.append(
                {
                    "participant": "",
                    metric: x_value - np.mean(x_values),
                    reaction_time_column: y_value - np.mean(y_values),
                }
            )
    return centered_rows


def analyze_metric_reaction_times(
    rows: Iterable[Mapping[str, object]],
    metrics: Sequence[str],
    *,
    reaction_time_column: str = "reaction_time",
    participant_column: str = "participant",
    min_trials: int = 3,
    include_pooled_within_participant: bool = True,
) -> list[dict[str, object]]:
    """Compute per-participant and pooled within-participant RT associations."""

    rows = list(rows)
    grouped_rows = _group_by_participant(rows, participant_column)
    summary_rows: list[dict[str, object]] = []
    for metric in metrics:
        for participant, participant_rows in grouped_rows.items():
            summary_rows.append(
                _association_row(
                    "participant",
                    participant,
                    metric,
                    participant_rows,
                    reaction_time_column=reaction_time_column,
                    min_trials=min_trials,
                )
            )
        if include_pooled_within_participant:
            centered_rows = _within_participant_centered_rows(grouped_rows, metric, reaction_time_column)
            summary_rows.append(
                _association_row(
                    "pooled_within_participant",
                    "",
                    metric,
                    centered_rows,
                    reaction_time_column=reaction_time_column,
                    min_trials=min_trials,
                )
            )
    return summary_rows
