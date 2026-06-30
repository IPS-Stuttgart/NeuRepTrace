"""Reject unsupported class/domain weighting options and guard small torch validation splits.

The installed shim also normalizes prediction-time feature validation for the
PyTorch-backed decoders so bad prediction inputs raise stable ValueErrors before
falling through to low-level tensor/matrix errors. It additionally validates the
Torch MLP estimator's scalar numeric options before the optional torch import so
misspecified configs fail deterministically even in lightweight environments.
"""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_neureptrace_torch_weight_validation_patch_installed"
_OPTIONAL_TORCH_IMPORT_ROOTS = ("torch", "pytorch_lightning", "lightning")
_ROW_STRATIFIED_FALLBACK_CLASSES = frozenset(
    {
        "TorchMLPClassifier",
        "TorchDANNClassifier",
        "TorchCDANClassifier",
    }
)


def _validate_weight_option(value: Any, *, name: str) -> None:
    """Accept only the documented torch decoder weighting modes."""

    if value is None or value == "balanced":
        return
    raise ValueError(f"{name} must be None or 'balanced'.")


def _integer_value(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(number) or number % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(number)


def _positive_integer_value(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(number) or number % 1.0 != 0.0:
        raise ValueError(f"{name} must be a positive integer.")
    integer = int(number)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _optional_integer_value(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _integer_value(value, name=name)


def _float_value(value: Any, *, name: str, expectation: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be {expectation}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {expectation}.") from exc
    return number


def _positive_float_value(value: Any, *, name: str) -> float:
    number = _float_value(value, name=name, expectation="positive and finite")
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _nonnegative_float_value(value: Any, *, name: str) -> float:
    number = _float_value(value, name=name, expectation="non-negative and finite")
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return number


def _bounded_float_value(value: Any, *, name: str, lower: float, upper: float) -> float:
    expectation = f"finite in [{lower}, {upper})"
    number = _float_value(value, name=name, expectation=expectation)
    if not np.isfinite(number) or number < lower or number >= upper:
        raise ValueError(f"{name} must be {expectation}.")
    return number


def _validate_torch_mlp_numeric_options(estimator: Any) -> None:
    """Reject ambiguous TorchMLPClassifier scalar options before importing torch."""

    _positive_integer_value(getattr(estimator, "hidden_units", 64), name="hidden_units")
    _positive_integer_value(getattr(estimator, "max_iter", 100), name="max_iter")
    _positive_integer_value(getattr(estimator, "batch_size", 128), name="batch_size")
    _positive_integer_value(getattr(estimator, "patience", 8), name="patience")
    _positive_float_value(getattr(estimator, "learning_rate", 1e-3), name="learning_rate")
    _nonnegative_float_value(getattr(estimator, "weight_decay", 1e-4), name="weight_decay")
    _bounded_float_value(getattr(estimator, "validation_fraction", 0.1), name="validation_fraction", lower=0.0, upper=1.0)
    _bounded_float_value(getattr(estimator, "dropout", 0.1), name="dropout", lower=0.0, upper=1.0)
    _optional_integer_value(getattr(estimator, "random_state", None), name="random_state")


def _valid_fraction(value: Any) -> float | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
        return None
    return fraction


def _labels_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    if len(args) >= 2:
        return args[1]
    return kwargs.get("labels", kwargs.get("y"))


def _label_vector(labels: Any) -> np.ndarray:
    """Return one label object per sample, preserving composite labels."""

    labels_array = np.asarray(labels, dtype=object)
    if labels_array.ndim == 0:
        vector = np.empty(1, dtype=object)
        vector[0] = labels_array.item()
        return vector
    if labels_array.ndim == 1:
        return labels_array.reshape(-1)

    rows = labels_array.reshape(labels_array.shape[0], -1)
    if rows.shape[1] == 1:
        return rows[:, 0].reshape(-1)
    vector = np.empty(rows.shape[0], dtype=object)
    for index, row in enumerate(rows):
        vector[index] = tuple(row.tolist())
    return vector


def _values_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _class_counts(labels: np.ndarray) -> list[int]:
    counts: list[int] = []
    classes: list[Any] = []
    for label in labels.tolist():
        for index, class_label in enumerate(classes):
            if _values_equal(label, class_label):
                counts[index] += 1
                break
        else:
            classes.append(label)
            counts.append(1)
    return counts


def _small_stratified_holdout(labels: Any, fraction_value: Any) -> bool:
    fraction = _valid_fraction(fraction_value)
    if fraction is None:
        return False
    labels_array = _label_vector(labels)
    if labels_array.size < 2:
        return True
    counts = _class_counts(labels_array)
    if len(counts) < 2 or min(counts) < 2:
        return True
    holdout_count = int(np.ceil(labels_array.size * fraction))
    return holdout_count < len(counts) or labels_array.size - holdout_count < len(counts)


def _prediction_feature_matrix(features: Any, *, n_features: Any, estimator_name: str) -> np.ndarray:
    """Return a validated prediction matrix for a fitted torch estimator."""

    try:
        expected_width = int(n_features)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive corrupt-state guard
        raise RuntimeError(f"{estimator_name} is missing its fitted feature width.") from exc
    if expected_width < 1:  # pragma: no cover - defensive corrupt-state guard
        raise RuntimeError(f"{estimator_name} has an invalid fitted feature width.")
    try:
        matrix = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{estimator_name} prediction features must be numeric.") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{estimator_name} prediction features must be a two-dimensional feature matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{estimator_name} prediction features must be finite.")
    if matrix.shape[1] != expected_width:
        raise ValueError(f"{estimator_name} prediction features width {matrix.shape[1]} does not match fitted width {expected_width}.")
    return matrix


def _missing_optional_torch_dependency(exc: ModuleNotFoundError) -> bool:
    """Return whether an import failed only because an optional torch extra is absent."""

    missing_name = getattr(exc, "name", "") or ""
    return any(missing_name == root or missing_name.startswith(f"{root}.") for root in _OPTIONAL_TORCH_IMPORT_ROOTS)


def _import_optional_torch_module(module_name: str) -> Any | None:
    """Import a torch-backed module without making optional extras mandatory at package import."""

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if _missing_optional_torch_dependency(exc):
            return None
        raise


def _install_fit_guard(
    class_object: type,
    *attribute_names: str,
    option_validator: Callable[[Any], None] | None = None,
) -> None:
    original_fit = class_object.fit
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, *args, **kwargs):
        if option_validator is not None:
            option_validator(self)
        for attribute_name in attribute_names:
            _validate_weight_option(getattr(self, attribute_name, None), name=attribute_name)
        labels = _labels_from_call(args, kwargs)
        validation_fraction = getattr(self, "validation_fraction", None)
        needs_training_loss_fallback = (
            class_object.__name__ in _ROW_STRATIFIED_FALLBACK_CLASSES
            and labels is not None
            and _small_stratified_holdout(labels, validation_fraction)
        )
        if not needs_training_loss_fallback:
            return original_fit(self, *args, **kwargs)
        self.validation_fraction = 0.0
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            self.validation_fraction = validation_fraction

    setattr(fit, _PATCH_MARKER, True)
    class_object.fit = fit


def _install_prediction_guard(class_object: type, method_name: str, *, estimator_name: str) -> None:
    original_method = getattr(class_object, method_name)
    if getattr(original_method, _PATCH_MARKER, False):
        return

    @wraps(original_method)
    def method(self, features, *args, **kwargs):
        if not hasattr(self, "model_"):
            return original_method(self, features, *args, **kwargs)
        matrix = _prediction_feature_matrix(
            features,
            n_features=getattr(self, "n_features_in_", None),
            estimator_name=estimator_name,
        )
        return original_method(self, matrix, *args, **kwargs)

    setattr(method, _PATCH_MARKER, True)
    setattr(class_object, method_name, method)


def _install_logits_guard(class_object: type, *, estimator_name: str) -> None:
    _install_prediction_guard(class_object, "_logits", estimator_name=estimator_name)


def install() -> None:
    """Install validation for torch-backed decoders without requiring optional extras."""

    decoding = importlib.import_module("neureptrace.decoding")
    _install_fit_guard(
        decoding.TorchMLPClassifier,
        "class_weight",
        option_validator=_validate_torch_mlp_numeric_options,
    )
    _install_prediction_guard(decoding.TorchMLPClassifier, "decision_function", estimator_name="TorchMLPClassifier")
    _install_prediction_guard(decoding.TorchMLPClassifier, "predict_proba", estimator_name="TorchMLPClassifier")

    dann = _import_optional_torch_module("neureptrace.decoding.dann")
    if dann is not None:
        _install_fit_guard(dann.TorchDANNClassifier, "class_weight")
        _install_logits_guard(dann.TorchDANNClassifier, estimator_name="TorchDANNClassifier")

    cdan = _import_optional_torch_module("neureptrace.decoding.cdan")
    if cdan is not None:
        _install_fit_guard(cdan.TorchCDANClassifier, "class_weight")
        _install_logits_guard(cdan.TorchCDANClassifier, estimator_name="TorchCDANClassifier")

    source_domain_generalization = _import_optional_torch_module("neureptrace.decoding.source_domain_generalization")
    if source_domain_generalization is not None:
        _install_fit_guard(
            source_domain_generalization.TorchSourceDomainGeneralizationClassifier,
            "class_weight",
            "domain_weight",
        )
        _install_logits_guard(
            source_domain_generalization.TorchSourceDomainGeneralizationClassifier,
            estimator_name="TorchSourceDomainGeneralizationClassifier",
        )

    source_vrex = _import_optional_torch_module("neureptrace.decoding.source_vrex")
    if source_vrex is not None:
        _install_fit_guard(source_vrex.TorchVRExClassifier, "class_weight")
        _install_logits_guard(source_vrex.TorchVRExClassifier, estimator_name="TorchVRExClassifier")


__all__ = ["install"]
