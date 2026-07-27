"""Validate VREx numeric hyperparameters, fit features, identifiers, and batch sampling."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_vrex_numeric_config_patch_installed"
_DANN_NUMERIC_PATCH_MARKER = "_neureptrace_dann_bool_array_numeric_config_patch_installed"
_SOURCE_VREX_FEATURE_PATCH_MARKER = "_neureptrace_source_vrex_finite_fit_feature_patch_installed"
_SOURCE_VREX_DOMAIN_BATCH_PATCH_MARKER = "_neureptrace_source_vrex_domain_batch_patch_installed"
_LINEAR_VREX_FEATURE_MATRIX_PATCH_MARKER = "_neureptrace_linear_vrex_feature_matrix_iterable_patch_installed"
_LINEAR_VREX_IDENTIFIER_PATCH_MARKER = "_neureptrace_linear_vrex_identifier_patch_installed"


def _is_boolean_scalar_like(value: Any) -> bool:
    """Return true for scalar or single-item array boolean config values."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if not isinstance(value, np.ndarray):
        return False
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    if array.size != 1:
        return False
    item = array.reshape(-1)[0]
    if isinstance(item, np.generic):
        item = item.item()
    return isinstance(item, (bool, np.bool_))


def _is_complex_scalar_like(value: Any) -> bool:
    """Return true for scalar or single-item array complex config values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if not isinstance(value, np.ndarray):
        return False
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    if array.size != 1:
        return False
    item = array.reshape(-1)[0]
    if isinstance(item, np.generic):
        item = item.item()
    return isinstance(item, (complex, np.complexfloating))


def _positive_int(value: Any, *, name: str) -> int:
    if _is_boolean_scalar_like(value) or _is_complex_scalar_like(value):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: Any, *, name: str) -> float:
    if _is_boolean_scalar_like(value) or _is_complex_scalar_like(value):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: Any, *, name: str) -> float:
    if _is_boolean_scalar_like(value) or _is_complex_scalar_like(value):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed


def _materialize_feature_values(value: Any) -> Any:
    """Materialize nested one-pass feature iterables before NumPy consumes them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_feature_values(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__array__"):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_feature_values(item) for item in value]


def _contains_boolean_feature(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean_feature(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if hasattr(value, "__array__"):
        try:
            return _contains_boolean_feature(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, Iterable):
        return any(_contains_boolean_feature(item) for item in value)
    return False


def _contains_complex_feature(value: Any) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex_feature(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (complex, np.complexfloating))
    if hasattr(value, "__array__"):
        try:
            return _contains_complex_feature(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, Iterable):
        return any(_contains_complex_feature(item) for item in value)
    return False


def _linear_vrex_feature_matrix(values: Any, *, name: str) -> np.ndarray:
    """Convert array-like or one-pass feature rows into a finite 2-D matrix."""

    raw_values = _materialize_feature_values(values)
    if _contains_boolean_feature(raw_values):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    if _contains_complex_feature(raw_values):
        raise ValueError(f"{name} must contain real-valued feature values, not complex values.")

    try:
        matrix = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty two-dimensional feature matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional feature matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


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


def _install_dann_numeric_validators() -> None:
    dann = importlib.import_module("neureptrace.decoding.dann")
    if getattr(dann._integer, _DANN_NUMERIC_PATCH_MARKER, False):
        return

    original_integer = dann._integer
    original_positive_float = dann._positive_float
    original_nonnegative_float = dann._nonnegative_float
    original_bounded_float = dann._bounded_float

    def _integer(value: Any, name: str) -> int:
        if _is_boolean_scalar_like(value):
            raise ValueError(f"{name} must be an integer.")
        return original_integer(value, name)

    def _positive_float_wrapper(value: Any, name: str) -> float:
        if _is_boolean_scalar_like(value):
            raise ValueError(f"{name} must be positive and finite.")
        return original_positive_float(value, name)

    def _nonnegative_float_wrapper(value: Any, name: str) -> float:
        if _is_boolean_scalar_like(value):
            raise ValueError(f"{name} must be non-negative and finite.")
        return original_nonnegative_float(value, name)

    def _bounded_float_wrapper(value: Any, name: str, *, lower: float, upper: float) -> float:
        if _is_boolean_scalar_like(value):
            raise ValueError(f"{name} must be finite in [{lower}, {upper}).")
        return original_bounded_float(value, name, lower=lower, upper=upper)

    for function in (_integer, _positive_float_wrapper, _nonnegative_float_wrapper, _bounded_float_wrapper):
        setattr(function, _DANN_NUMERIC_PATCH_MARKER, True)

    dann._integer = _integer
    dann._positive_float = _positive_float_wrapper
    dann._nonnegative_float = _nonnegative_float_wrapper
    dann._bounded_float = _bounded_float_wrapper


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


def _install_linear_vrex_feature_matrix() -> None:
    vrex = importlib.import_module("neureptrace.decoding.vrex")
    if getattr(vrex._feature_matrix, _LINEAR_VREX_FEATURE_MATRIX_PATCH_MARKER, False):
        return

    setattr(_linear_vrex_feature_matrix, _LINEAR_VREX_FEATURE_MATRIX_PATCH_MARKER, True)
    vrex._feature_matrix = _linear_vrex_feature_matrix


def _install_linear_vrex_identifier_encoding() -> None:
    """Encode labels/domains with missing-aware equality before fitting VREx."""

    from neureptrace.decoding._domain_ids import ordered_unique, values_equal

    vrex = importlib.import_module("neureptrace.decoding.vrex")
    original_fit = vrex.LinearVRExClassifier.fit
    if getattr(original_fit, _LINEAR_VREX_IDENTIFIER_PATCH_MARKER, False):
        return

    def _encode(values: np.ndarray, unique_values: tuple[object, ...]) -> np.ndarray:
        return np.asarray(
            [
                next(index for index, unique_value in enumerate(unique_values) if values_equal(value, unique_value))
                for value in values.tolist()
            ],
            dtype=int,
        )

    def _encoded_class_weight(class_weight: Mapping[Any, Any], classes: tuple[object, ...]) -> dict[int, Any]:
        encoded: dict[int, Any] = {}
        entries = tuple(class_weight.items())
        for index, class_label in enumerate(classes):
            encoded[index] = next(
                (weight for key, weight in entries if values_equal(key, class_label)),
                1.0,
            )
        return encoded

    @wraps(original_fit)
    def fit(self, source_features, source_labels, *, source_domains):
        features = vrex._feature_matrix(source_features, name="source_features")
        labels = vrex._object_vector(source_labels, expected_length=features.shape[0], name="source_labels")
        domains = vrex._object_vector(source_domains, expected_length=features.shape[0], name="source_domains")
        vrex._validate_hashable(domains, name="source_domains")

        classes = ordered_unique(labels)
        domain_values = ordered_unique(domains)
        encoded_labels = _encode(labels, classes)
        encoded_domains = _encode(domains, domain_values)

        original_class_weight = self.class_weight
        if isinstance(original_class_weight, Mapping):
            self.class_weight = _encoded_class_weight(original_class_weight, classes)
        try:
            result = original_fit(
                self,
                features,
                encoded_labels,
                source_domains=encoded_domains,
            )
        finally:
            self.class_weight = original_class_weight

        self.classes_ = vrex._object_vector(classes, expected_length=len(classes), name="classes")
        self.source_domains_ = vrex._object_vector(
            domain_values,
            expected_length=len(domain_values),
            name="source_domains",
        )
        return result

    setattr(fit, _LINEAR_VREX_IDENTIFIER_PATCH_MARKER, True)
    vrex.LinearVRExClassifier.fit = fit


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
    """Install VREx hyperparameter, fit-input, identifier, and batch validators."""

    _install_dann_numeric_validators()
    _install_linear_vrex_numeric_validators()
    _install_linear_vrex_feature_matrix()
    _install_linear_vrex_identifier_encoding()
    _install_source_vrex_fit_feature_validator()
    _install_source_vrex_domain_balanced_batch()


__all__ = ["install"]
