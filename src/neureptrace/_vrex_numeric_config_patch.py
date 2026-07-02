"""Validate VREx numeric hyperparameters, fit features, and batch sampling."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_vrex_numeric_config_patch_installed"
_SOURCE_VREX_FEATURE_PATCH_MARKER = "_neureptrace_source_vrex_finite_fit_feature_patch_installed"
_SOURCE_VREX_DOMAIN_BATCH_PATCH_MARKER = "_neureptrace_source_vrex_domain_batch_patch_installed"


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed


def _validate_finite_source_features(source_features: Any) -> None:
    """Reject NaN/Inf VREx fit features before torch training starts."""

    try:
        matrix = np.asarray(source_features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("vrex source_features must be numeric and finite.") from exc
    if matrix.ndim == 2 and not np.all(np.isfinite(matrix)):
        raise ValueError("vrex source_features must contain finite values.")


def _domain_balanced_batch(train_idx: np.ndarray, domains: np.ndarray, *, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a batch across domains present in the training split."""

    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    domains = np.asarray(domains).reshape(-1)
    if train_idx.size == 0:
        raise ValueError("train_idx must contain at least one row for VREx domain-balanced batching.")
    if np.any(train_idx < 0) or np.any(train_idx >= domains.shape[0]):
        raise ValueError("train_idx contains indices outside source_domains.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive for VREx domain-balanced batching.")

    domain_values = np.unique(domains[train_idx])
    per_domain = max(1, int(np.ceil(int(batch_size) / domain_values.shape[0])))
    chunks = []
    for domain in domain_values:
        candidates = train_idx[domains[train_idx] == domain]
        chunks.append(rng.choice(candidates, size=min(per_domain, int(batch_size)), replace=candidates.shape[0] < per_domain))
    batch = np.concatenate(chunks)
    rng.shuffle(batch)
    return batch[: int(batch_size)]


def _install_linear_vrex_numeric_validators() -> None:
    vrex = importlib.import_module("neureptrace.decoding.vrex")
    if getattr(vrex._positive_int, _PATCH_MARKER, False):
        return

    setattr(_positive_int, _PATCH_MARKER, True)
    setattr(_positive_float, _PATCH_MARKER, True)
    setattr(_nonnegative_float, _PATCH_MARKER, True)
    vrex._positive_int = _positive_int
    vrex._positive_float = _positive_float
    vrex._nonnegative_float = _nonnegative_float


def _install_source_vrex_fit_feature_validator() -> None:
    source_vrex = importlib.import_module("neureptrace.decoding.source_vrex")
    original_fit = source_vrex.TorchVRExClassifier.fit
    if getattr(original_fit, _SOURCE_VREX_FEATURE_PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, source_features, source_labels, *, source_domains):
        _validate_finite_source_features(source_features)
        return original_fit(self, source_features, source_labels, source_domains=source_domains)

    setattr(fit, _SOURCE_VREX_FEATURE_PATCH_MARKER, True)
    source_vrex.TorchVRExClassifier.fit = fit


def _install_source_vrex_domain_balanced_batch() -> None:
    source_vrex = importlib.import_module("neureptrace.decoding.source_vrex")
    if getattr(source_vrex._domain_balanced_batch, _SOURCE_VREX_DOMAIN_BATCH_PATCH_MARKER, False):
        return

    setattr(_domain_balanced_batch, _SOURCE_VREX_DOMAIN_BATCH_PATCH_MARKER, True)
    source_vrex._domain_balanced_batch = _domain_balanced_batch


def install() -> None:
    """Install VREx hyperparameter, fit-feature, and batch-sampling validators."""

    _install_linear_vrex_numeric_validators()
    _install_source_vrex_fit_feature_validator()
    _install_source_vrex_domain_balanced_batch()


__all__ = ["install"]
