"""Runtime patch for exact observation-ensemble label and rank handling."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from ._response_window_bool_numeric_patch import (
    _exact_integer_labels,
    _numeric_probability_labels,
)

_LABEL_PATCH_MARKER = (
    "_neureptrace_observation_ensemble_exact_label_patch_installed"
)
_LABEL_PATCH_VERSION = 3
_WRAPPER_VERSION_ATTR = "_observation_ensemble_exact_label_patch_version"


def _label_values(prob_columns: Sequence[str]) -> tuple[int, ...]:
    """Return exact signed probability labels, or positions for named classes."""

    labels = _numeric_probability_labels(prob_columns)
    if labels is None:
        return tuple(range(len(prob_columns)))
    return labels


def _integer_label_values(
    labels: Sequence[object] | np.ndarray | pd.Series,
    *,
    column_name: str = "true_label",
) -> np.ndarray:
    """Parse signed-64-bit labels without a lossy float64 round-trip."""

    try:
        return _exact_integer_labels(labels, label_name=column_name)
    except ValueError as exc:
        # Preserve the established observation-ensemble error contract for
        # fractional class labels while using the shared exact parser for all
        # other validation and range checks.
        if "must be integer-valued." not in str(exc):
            raise
        values = pd.Series(labels).tolist()
        invalid: list[object] = []
        for value in values:
            try:
                _exact_integer_labels([value], label_name=column_name)
            except ValueError as item_exc:
                if "must be integer-valued." in str(item_exc):
                    invalid.append(value)
            if len(invalid) >= 5:
                break
        raise ValueError(
            f"{column_name} values must be integer-valued class labels; "
            f"invalid values: {invalid}"
        ) from exc


def _rank_scores(probabilities: np.ndarray) -> np.ndarray:
    """Return normalized average ranks without ordering exact ties by class."""

    n_classes = probabilities.shape[1]
    if n_classes <= 1:
        return np.zeros_like(probabilities, dtype=float)

    scores = np.empty_like(probabilities, dtype=float)
    for row_index, row in enumerate(probabilities):
        order = np.argsort(row, kind="stable")
        sorted_values = row[order]
        sorted_scores = np.arange(n_classes, dtype=float)
        group_start = 0
        for group_stop in range(1, n_classes + 1):
            if (
                group_stop < n_classes
                and sorted_values[group_stop] == sorted_values[group_start]
            ):
                continue
            sorted_scores[group_start:group_stop] = 0.5 * (
                group_start + group_stop - 1
            )
            group_start = group_stop
        scores[row_index, order] = sorted_scores
    return scores / float(n_classes - 1)


def _normalize_exact_label_outputs(
    output: pd.DataFrame,
    observation_ensemble: Any,
) -> pd.DataFrame:
    """Recompute label-derived columns from exact probability-column labels."""

    prob_columns = observation_ensemble.probability_columns(output)
    if not prob_columns:
        return output

    label_values = _label_values(prob_columns)
    labels = np.asarray(label_values, dtype=np.int64)
    probabilities = output.loc[:, list(prob_columns)].to_numpy(dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        return output

    normalized = output.copy()
    predicted_labels = labels[np.argmax(probabilities, axis=1)]
    normalized["predicted_label"] = predicted_labels

    if "true_label" not in normalized.columns:
        return normalized

    true_labels = _integer_label_values(normalized["true_label"])
    label_to_position = {
        int(label): position for position, label in enumerate(label_values)
    }
    positions = np.asarray(
        [label_to_position.get(int(label), -1) for label in true_labels],
        dtype=int,
    )
    if bool((positions < 0).any()):
        missing = sorted(
            {
                int(label)
                for label, position in zip(true_labels, positions, strict=True)
                if position < 0
            }
        )
        raise ValueError(
            "true_label values must index probability labels "
            f"{list(label_values)}; missing labels: {missing[:5]}"
        )

    row_indices = np.arange(len(normalized), dtype=int)
    normalized["probability_true_class"] = probabilities[
        row_indices,
        positions,
    ]
    normalized["is_correct"] = predicted_labels == true_labels
    return normalized


def _refresh_helper_aliases(observation_ensemble: Any) -> None:
    """Install helpers on both the public package and its delegated implementation."""

    modules = [observation_ensemble]
    implementation = getattr(observation_ensemble, "_impl", None)
    if implementation is not None:
        modules.append(implementation)

    for module in modules:
        module._label_values = _label_values
        module._integer_label_values = _integer_label_values
        module._rank_scores = _rank_scores


def install() -> None:
    """Preserve exact labels, tied ranks, and probability-label coverage."""

    import neureptrace.observation_ensemble as observation_ensemble

    # The public compatibility package copies helpers from a separately loaded
    # implementation module. Refresh both namespaces so the delegated ensemble
    # function uses the corrected helpers as well.
    _refresh_helper_aliases(observation_ensemble)
    setattr(
        observation_ensemble,
        _LABEL_PATCH_MARKER,
        _LABEL_PATCH_VERSION,
    )

    current = observation_ensemble.ensemble_probability_observations
    if getattr(current, _WRAPPER_VERSION_ATTR, 0) >= _LABEL_PATCH_VERSION:
        return

    @wraps(current)
    def ensemble_probability_observations(*args: Any, **kwargs: Any):
        output = current(*args, **kwargs)
        return _normalize_exact_label_outputs(output, observation_ensemble)

    ensemble_probability_observations._missing_label_patch = True  # type: ignore[attr-defined]
    setattr(
        ensemble_probability_observations,
        _WRAPPER_VERSION_ATTR,
        _LABEL_PATCH_VERSION,
    )
    observation_ensemble.ensemble_probability_observations = (
        ensemble_probability_observations
    )


__all__ = ["install"]
