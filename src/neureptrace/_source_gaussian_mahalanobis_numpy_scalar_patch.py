"""Patch Gaussian/Mahalanobis source-decoder numeric validation edge cases.

This module keeps strict source-only Gaussian and Mahalanobis helpers robust for
configuration scalars, feature-matrix validation, and Gaussian result precision.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_INSTALLED = False


def _numeric_scalar(value: Any, *, name: str, allow_zero: bool) -> float:
    kind = "non-negative" if allow_zero else "positive"
    message = f"{name} must be {kind} and finite."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return _numeric_scalar(value.item(), name=name, allow_zero=allow_zero)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed < 0.0 or (not allow_zero and parsed <= 0.0):
        raise ValueError(message)
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    return _numeric_scalar(value, name=name, allow_zero=False)


def _nonnegative_float(value: Any, *, name: str) -> float:
    return _numeric_scalar(value, name=name, allow_zero=True)


def _contains_boolean_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, Sequence):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _has_boolean_feature_values(values: Any) -> bool:
    if _contains_boolean_value(values):
        return True
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    if array.dtype == np.bool_:
        return True
    if array.dtype == object:
        return _contains_boolean_value(array)
    return False


def _reject_boolean_feature_values(values: Any, *, name: str) -> None:
    if _has_boolean_feature_values(values):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion keeps every finite nonzero value usable."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return values
    if np.any((values != 0.0) & (compact == 0.0)):
        return values
    return compact


def install() -> None:
    """Install scalar, feature, and Gaussian result-precision guards."""

    global _INSTALLED
    if _INSTALLED:
        return

    importlib.import_module("neureptrace._source_interpolation_one_pass_patch").install()

    from neureptrace.decoding import source_gaussian, source_mahalanobis

    original_gaussian_feature_matrix = source_gaussian._feature_matrix
    original_mahalanobis_feature_matrix = source_mahalanobis._feature_matrix
    original_fit_source_gaussian_decoder = source_gaussian.fit_source_gaussian_decoder

    @wraps(original_gaussian_feature_matrix)
    def _gaussian_feature_matrix(values: Any, *, name: str) -> np.ndarray:
        _reject_boolean_feature_values(values, name=name)
        return original_gaussian_feature_matrix(values, name=name)

    @wraps(original_mahalanobis_feature_matrix)
    def _mahalanobis_feature_matrix(values: Any, *, name: str) -> np.ndarray:
        _reject_boolean_feature_values(values, name=name)
        return original_mahalanobis_feature_matrix(values, name=name)

    @wraps(original_fit_source_gaussian_decoder)
    def _fit_source_gaussian_decoder(
        *,
        source_features: Any,
        source_labels: Any,
        test_features: Any,
        config: Any = None,
    ) -> Any:
        cfg = source_gaussian.source_gaussian_config() if config is None else source_gaussian._coerce_config(config)
        source = source_gaussian._feature_matrix(source_features, name="source_features")
        test = source_gaussian._feature_matrix(test_features, name="test_features")
        if source.shape[1] != test.shape[1]:
            raise ValueError(
                "source_features and test_features must have the same feature width: "
                f"{source.shape[1]} != {test.shape[1]}."
            )
        labels = source_gaussian._label_vector(
            source_labels,
            expected_length=source.shape[0],
            name="source_labels",
        )
        classes, _ = source_gaussian.label_counts(labels)
        if classes.shape[0] < 2:
            raise ValueError("At least two source classes are required.")

        means, class_variances, counts = source_gaussian._class_gaussian_stats(
            source,
            labels,
            classes=classes,
            variance_floor=cfg.variance_floor,
        )
        variances = source_gaussian._apply_covariance_type(
            class_variances,
            counts=counts,
            covariance_type=cfg.covariance_type,
            variance_floor=cfg.variance_floor,
        )
        priors = source_gaussian._class_priors(counts, mode=cfg.prior)
        log_likelihoods = source_gaussian.gaussian_log_likelihoods(
            test,
            means=means,
            variances=variances,
        )
        log_posteriors = (log_likelihoods + np.log(priors)[None, :]) / cfg.temperature
        probabilities = source_gaussian._softmax(log_posteriors)
        predictions = classes[np.argmax(probabilities, axis=1)]
        metadata = source_gaussian._metadata(
            cfg,
            n_source_rows=source.shape[0],
            n_test_rows=test.shape[0],
            feature_dim=source.shape[1],
            classes=classes,
            counts=counts,
        )
        return source_gaussian.SourceGaussianResult(
            probabilities=probabilities.astype(np.float32, copy=False),
            predictions=predictions,
            classes=classes,
            means=_compact_float32(means),
            variances=_compact_float32(variances),
            priors=priors.astype(np.float32, copy=False),
            log_likelihoods=_compact_float32(log_likelihoods),
            metadata=metadata,
        )

    source_gaussian._positive_float = _positive_float
    source_gaussian._feature_matrix = _gaussian_feature_matrix
    source_gaussian.fit_source_gaussian_decoder = _fit_source_gaussian_decoder
    source_mahalanobis._positive_float = _positive_float
    source_mahalanobis._nonnegative_float = _nonnegative_float
    source_mahalanobis._feature_matrix = _mahalanobis_feature_matrix
    _INSTALLED = True


__all__ = ["install"]
