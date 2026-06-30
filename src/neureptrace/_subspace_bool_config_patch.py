"""Normalize subspace and random-subspace config values from CLI/YAML inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_subspace_bool_config_patch_installed"
_RANDOM_SUBSPACE_PATCH_MARKER = "_neureptrace_random_subspace_bool_config_patch_installed"
_RANDOM_SUBSPACE_RANDOM_STATE_PATCH_MARKER = "_neureptrace_random_subspace_random_state_patch_installed"
_RANDOM_SUBSPACE_DECISION_ALIGNMENT_PATCH_MARKER = "_neureptrace_random_subspace_decision_alignment_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_NONE_STRINGS = {"", "none", "null"}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a real bool while rejecting ambiguous truthy/falsy objects."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise _bool_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _bool_error(name)
        return _normalize_bool(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise _bool_error(name)
    raise _bool_error(name)


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer or None.")


def _normalize_optional_random_state(value: Any, *, name: str) -> int | None:
    """Normalize optional integer seeds without leaking raw set/NumPy errors."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _NONE_STRINGS:
            return None
        value = stripped
    elif isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        return _normalize_optional_random_state(value.item(), name=name)
    elif isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        raise _random_state_error(name)
    if isinstance(value, (bool, np.bool_)):
        raise _random_state_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _random_state_error(name) from exc
    if not np.isfinite(parsed) or parsed < 0.0 or parsed % 1.0 != 0.0:
        raise _random_state_error(name)
    return int(parsed)


def _normalize_random_subspace_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "random_state" not in kwargs:
        return kwargs
    normalized = dict(kwargs)
    normalized["random_state"] = _normalize_optional_random_state(normalized["random_state"], name="random_state")
    return normalized


def _aligned_decision_probabilities(random_subspace: Any, model: Any, features: Any, *, classes: np.ndarray, epsilon: float) -> np.ndarray:
    """Convert decision-function scores onto the full ensemble class axis."""

    feature_rows = np.asarray(features).shape[0]
    scores = np.asarray(model.decision_function(features), dtype=float)
    model_classes = np.asarray(getattr(model, "classes_", classes), dtype=object).reshape(-1)
    if scores.ndim == 1:
        if model_classes.shape[0] != 2:
            raise ValueError("A one-dimensional decision_function output requires exactly two model classes.")
        scores = np.column_stack([-scores, scores])
    elif scores.ndim != 2:
        raise ValueError("decision_function must return a one- or two-dimensional score array.")
    if scores.shape[0] != feature_rows:
        raise ValueError(f"decision_function returned {scores.shape[0]} rows for {feature_rows} feature rows.")
    if scores.shape[1] != model_classes.shape[0]:
        raise ValueError(
            "decision_function column count must match the fitted estimator's classes_: "
            f"{scores.shape[1]} != {model_classes.shape[0]}."
        )

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    local_probabilities = random_subspace._normalize_probability_rows(np.exp(np.clip(shifted, -50.0, 50.0)), epsilon=epsilon)
    target_classes = np.asarray(classes, dtype=object).reshape(-1)
    aligned = np.full((feature_rows, target_classes.shape[0]), epsilon, dtype=float)
    for source_column, class_label in enumerate(model_classes.tolist()):
        for target_column, target_label in enumerate(target_classes.tolist()):
            if random_subspace._labels_equal(class_label, target_label):
                aligned[:, target_column] = local_probabilities[:, source_column]
                break
    return random_subspace._normalize_probability_rows(aligned, epsilon=epsilon)


def install() -> None:
    """Install strict normalization for subspace and random-subspace config."""

    from neureptrace.decoding import subspace_adaptation as subspace

    original_config = subspace.subspace_adaptation_config
    if not getattr(original_config, _PATCH_MARKER, False):
        @wraps(original_config)
        def subspace_adaptation_config(
            *,
            method: str | None = subspace.DEFAULT_SUBSPACE_METHOD,
            n_components: int | str | None = subspace.DEFAULT_SUBSPACE_COMPONENTS,
            regularization: float | str = subspace.DEFAULT_SUBSPACE_REGULARIZATION,
            eigen_ridge: float | str = subspace.DEFAULT_SUBSPACE_EIGEN_RIDGE,
            standardize: Any = True,
            class_balance_source: Any = False,
            normalize_latent: Any = False,
        ):
            normalized_method = subspace.normalize_subspace_method(method)
            requested_balance = _normalize_bool(class_balance_source, name="class_balance_source")
            return original_config(
                method=normalized_method,
                n_components=n_components,
                regularization=regularization,
                eigen_ridge=eigen_ridge,
                standardize=_normalize_bool(standardize, name="standardize"),
                class_balance_source=(requested_balance or normalized_method == "balanced_tca"),
                normalize_latent=_normalize_bool(normalize_latent, name="normalize_latent"),
            )

        setattr(subspace_adaptation_config, _PATCH_MARKER, True)
        subspace.subspace_adaptation_config = subspace_adaptation_config

    from neureptrace.decoding import random_subspace

    original_random_subspace_config = random_subspace.random_subspace_ensemble_config
    if not getattr(original_random_subspace_config, _RANDOM_SUBSPACE_PATCH_MARKER, False):
        @wraps(original_random_subspace_config)
        def random_subspace_ensemble_config(**kwargs):
            kwargs = _normalize_random_subspace_kwargs(dict(kwargs))
            if "bootstrap_rows" in kwargs:
                kwargs["bootstrap_rows"] = _normalize_bool(kwargs["bootstrap_rows"], name="bootstrap_rows")
            return original_random_subspace_config(**kwargs)

        setattr(random_subspace_ensemble_config, _RANDOM_SUBSPACE_PATCH_MARKER, True)
        random_subspace.random_subspace_ensemble_config = random_subspace_ensemble_config

    original_sample_feature_subspaces = random_subspace.sample_feature_subspaces
    if not getattr(original_sample_feature_subspaces, _RANDOM_SUBSPACE_RANDOM_STATE_PATCH_MARKER, False):
        @wraps(original_sample_feature_subspaces)
        def sample_feature_subspaces(**kwargs):
            kwargs = _normalize_random_subspace_kwargs(dict(kwargs))
            return original_sample_feature_subspaces(**kwargs)

        setattr(sample_feature_subspaces, _RANDOM_SUBSPACE_RANDOM_STATE_PATCH_MARKER, True)
        random_subspace.sample_feature_subspaces = sample_feature_subspaces

    original_aligned_probabilities = random_subspace._aligned_probabilities
    if not getattr(original_aligned_probabilities, _RANDOM_SUBSPACE_DECISION_ALIGNMENT_PATCH_MARKER, False):
        @wraps(original_aligned_probabilities)
        def _aligned_probabilities(model, features, *, classes, epsilon):
            if not hasattr(model, "predict_proba") and hasattr(model, "decision_function"):
                return _aligned_decision_probabilities(random_subspace, model, features, classes=classes, epsilon=epsilon)
            return original_aligned_probabilities(model, features, classes=classes, epsilon=epsilon)

        setattr(_aligned_probabilities, _RANDOM_SUBSPACE_DECISION_ALIGNMENT_PATCH_MARKER, True)
        random_subspace._aligned_probabilities = _aligned_probabilities


__all__ = ["install"]
