"""Runtime patch for exact observation-ensemble label handling."""

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
_LABEL_PATCH_VERSION = 2


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


def _missing_true_labels(
    output: pd.DataFrame,
    observation_ensemble: Any,
) -> list[int]:
    prob_columns = observation_ensemble.probability_columns(output)
    if (
        "true_label" not in output.columns
        or "probability_true_class" not in output.columns
        or not prob_columns
    ):
        return []

    true_probability = pd.to_numeric(
        output["probability_true_class"], errors="coerce"
    )
    if not bool(true_probability.isna().any()):
        return []

    label_values = observation_ensemble._label_values(prob_columns)
    valid_labels = {int(label) for label in label_values}
    true_labels = observation_ensemble._integer_label_values(
        output["true_label"]
    )
    return sorted(
        {
            int(label)
            for label in true_labels
            if int(label) not in valid_labels
        }
    )


def install() -> None:
    """Preserve exact labels and reject labels absent from probability columns."""

    import neureptrace.observation_ensemble as observation_ensemble

    # Always refresh these aliases. Older runtime-patch installations can leave
    # the module-level marker set while the helpers still point at the legacy
    # float64/unsigned-suffix implementations.
    observation_ensemble._label_values = _label_values
    observation_ensemble._integer_label_values = _integer_label_values
    setattr(
        observation_ensemble,
        _LABEL_PATCH_MARKER,
        _LABEL_PATCH_VERSION,
    )

    if getattr(
        observation_ensemble.ensemble_probability_observations,
        "_missing_label_patch",
        False,
    ):
        return

    original_ensemble_probability_observations = (
        observation_ensemble.ensemble_probability_observations
    )

    @wraps(original_ensemble_probability_observations)
    def ensemble_probability_observations(*args: Any, **kwargs: Any):
        output = original_ensemble_probability_observations(*args, **kwargs)
        missing = _missing_true_labels(output, observation_ensemble)
        if missing:
            prob_columns = observation_ensemble.probability_columns(output)
            label_values = observation_ensemble._label_values(prob_columns)
            raise ValueError(
                "true_label values must index probability labels "
                f"{list(label_values)}; missing labels: {missing[:5]}"
            )
        return output

    ensemble_probability_observations._missing_label_patch = True  # type: ignore[attr-defined]
    observation_ensemble.ensemble_probability_observations = (
        ensemble_probability_observations
    )


__all__ = ["install"]
