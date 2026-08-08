"""Restore validation guarantees after runtime patch composition."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_patch_composition_regression_fix_installed"
_CLASS_SCORE_MARKER = "_neureptrace_composed_class_score_guard_installed"
_INTEGER_VALIDATION_MARKER = "_neureptrace_composed_integer_validation_guard_installed"
_STIMULUS_INSTALL_MARKER = "_neureptrace_safe_stimulus_annotation_install_installed"


def _materialize_nested(values: object) -> object:
    """Materialize one-pass nested iterables without changing reusable arrays."""

    if isinstance(values, np.ndarray) or isinstance(values, (str, bytes)):
        return values
    if hasattr(values, "__array__") or not isinstance(values, Iterable):
        return values
    return [_materialize_nested(value) for value in values]


def _contains_boolean(values: object) -> bool:
    """Return whether an arbitrarily nested score output contains booleans."""

    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return bool(values.size)
        if values.dtype == object:
            return any(_contains_boolean(value) for value in values.ravel(order="C"))
        return False
    if hasattr(values, "__array__"):
        try:
            return _contains_boolean(np.asarray(values, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        return False
    return any(_contains_boolean(value) for value in values)


def _install_class_score_guards() -> None:
    """Reapply score-domain guards after the composite-label wrapper installs."""

    from neureptrace.decoding import class_scores

    original = class_scores.as_class_score_matrix
    if getattr(original, _CLASS_SCORE_MARKER, False):
        return

    @wraps(original)
    def as_class_score_matrix(
        raw_scores: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
        classes: Sequence | np.ndarray,
        *,
        n_samples: int,
    ) -> np.ndarray | None:
        materialized = _materialize_nested(raw_scores)
        if _contains_boolean(materialized):
            raise ValueError("raw_scores must contain numeric score values, not boolean flags.")
        matrix = original(materialized, classes, n_samples=n_samples)
        if matrix is not None and not np.all(np.isfinite(matrix)):
            raise ValueError("raw_scores must contain only finite values.")
        return matrix

    setattr(as_class_score_matrix, _CLASS_SCORE_MARKER, True)
    as_class_score_matrix.__wrapped__ = original
    class_scores.as_class_score_matrix = as_class_score_matrix


def _install_integer_validation_messages() -> None:
    """Keep decoded-fold validation semantics and public error contracts aligned."""

    from neureptrace import _observation_schema_probability_patch as probability_patch

    original = probability_patch._validate_decoded_fold_integer_values
    if getattr(original, _INTEGER_VALIDATION_MARKER, False):
        return

    @wraps(original)
    def validate(values: Any, *, name: str) -> None:
        try:
            original(values, name=name)
        except ValueError as exc:
            message = str(exc)
            message = message.replace(
                f"from_decoded_fold {name} must be integer-valued, not boolean.",
                f"from_decoded_fold {name} must contain integer values, not boolean flags; inputs must be integer-valued.",
            )
            message = message.replace(
                f"from_decoded_fold {name} must be integer-valued.",
                f"from_decoded_fold {name} must contain finite integer values; inputs must be integer-valued.",
            )
            raise ValueError(message) from exc

    setattr(validate, _INTEGER_VALIDATION_MARKER, True)
    validate.__wrapped__ = original
    probability_patch._validate_decoded_fold_integer_values = validate


def _install_safe_stimulus_annotation_patch() -> None:
    """Skip the optional annotation patch when a lightweight public module is used."""

    from neureptrace import _stimulus_annotation_index_patch as annotation_patch

    original = annotation_patch.install
    if getattr(original, _STIMULUS_INSTALL_MARKER, False):
        return

    required_symbols = (
        annotation_patch._MATCH_NAME,
        "_add_annotation_candidate_columns",
        "_annotation_id",
        "_annotation_match_key",
        "_annotation_value",
        "_stream_columns",
    )

    @wraps(original)
    def install() -> None:
        public_module = importlib.import_module(annotation_patch._PUBLIC_MODULE)
        if any(symbol not in public_module.__dict__ for symbol in required_symbols):
            return
        original()

    setattr(install, _STIMULUS_INSTALL_MARKER, True)
    install.__wrapped__ = original
    annotation_patch.install = install


def install() -> None:
    """Install the compatibility repairs once, after all regular runtime patches."""

    import neureptrace

    if getattr(neureptrace, _PATCH_MARKER, False):
        return
    _install_class_score_guards()
    _install_integer_validation_messages()
    _install_safe_stimulus_annotation_patch()
    setattr(neureptrace, _PATCH_MARKER, True)


__all__ = ["install"]
