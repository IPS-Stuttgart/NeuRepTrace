"""Transfer Component Analysis utilities for cross-subject decoding.

The helpers in this module implement a dependency-light, protocol-explicit
Transfer Component Analysis (TCA) feature mapper.  TCA learns a shared latent
space from source rows and unlabeled held-out target rows by minimizing a marginal
source-target discrepancy while preserving variance.  A downstream classifier can
then be trained with source labels in that latent space.

This is a Category-2 protocol: source labels may be used by the classifier,
unlabeled target features are used by the representation learner, and held-out
target labels are not accepted by the public API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

TRANSFER_COMPONENT_PROTOCOL = "unlabeled_target_transfer_component_analysis"
TRANSFER_COMPONENT_CATEGORY = "2_unlabeled_target_adaptive"
TRANSFER_COMPONENT_KERNELS = ("linear", "rbf")
TRANSFER_COMPONENT_STANDARDIZE_SCOPES = ("source", "source_target", "none")
DEFAULT_TRANSFER_COMPONENTS = 16
DEFAULT_TCA_REGULARIZATION = 1e-3
DEFAULT_TCA_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class TransferComponentConfig:
    """Configuration for Transfer Component Analysis."""

    n_components: int | str = DEFAULT_TRANSFER_COMPONENTS
    kernel: str = "linear"
    regularization: float = DEFAULT_TCA_REGULARIZATION
    gamma: float | str | None = "scale"
    standardize_scope: str = "source"
    center_kernel: bool = True
    epsilon: float = DEFAULT_TCA_EPSILON


@dataclass(frozen=True, slots=True)
class TransferComponentResult:
    """Source and target features transformed into a TCA latent space."""

    source_features: np.ndarray
    target_features: np.ndarray
    config: TransferComponentConfig
    projection: np.ndarray
    eigenvalues: tuple[float, ...]
    source_mean: np.ndarray
    source_scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransferComponentClassificationResult:
    """Classifier outputs in a TCA latent space."""

    source_features: np.ndarray
    target_features: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray
    classifier: BaseEstimator
    tca: TransferComponentResult
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def transfer_component_config(
    *,
    n_components: int | str = DEFAULT_TRANSFER_COMPONENTS,
    kernel: str | None = "linear",
    regularization: float | str = DEFAULT_TCA_REGULARIZATION,
    gamma: float | str | None = "scale",
    standardize_scope: str | None = "source",
    center_kernel: bool = True,
    epsilon: float | str = DEFAULT_TCA_EPSILON,
) -> TransferComponentConfig:
    """Normalize user-facing TCA options."""

    return TransferComponentConfig(
        n_components=_normalize_components(n_components),
        kernel=normalize_transfer_component_kernel(kernel),
        regularization=_nonnegative_float(regularization, name="regularization"),
        gamma=gamma,
        standardize_scope=normalize_standardize_scope(standardize_scope),
        center_kernel=bool(center_kernel),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_transfer_component_features(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    config: TransferComponentConfig | Mapping[str, Any] | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
) -> TransferComponentResult:
    """Fit TCA from source rows plus unlabeled target rows.

    ``target_labels`` is deliberately rejected to keep the protocol Category 2.
    """

    if target_labels is not None:
        raise ValueError("Transfer Component Analysis does not accept target_labels; target labels must be reserved for scoring.")
    cfg = _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")

    source_scaled, target_scaled, mean, scale = _standardize_source_target(source, target, scope=cfg.standardize_scope)
    combined = np.vstack([source_scaled, target_scaled]).astype(float, copy=False)
    n_source = source.shape[0]
    n_target = target.shape[0]
    n_total = combined.shape[0]
    components = _effective_components(cfg.n_components, max_components=n_total - 1 if cfg.kernel == "rbf" else min(combined.shape[1], n_total - 1))

    mmd = _mmd_matrix(n_source, n_target)
    centering = np.eye(n_total, dtype=float) - np.full((n_total, n_total), 1.0 / n_total)

    if cfg.kernel == "linear":
        projection, eigenvalues = _fit_linear_tca(combined, mmd=mmd, centering=centering, cfg=cfg, n_components=components)
        embedded = combined @ projection
    else:
        kernel_matrix = _kernel_matrix(combined, kernel=cfg.kernel, gamma=cfg.gamma)
        if cfg.center_kernel:
            kernel_matrix = centering @ kernel_matrix @ centering
        projection, eigenvalues = _fit_kernel_tca(kernel_matrix, mmd=mmd, centering=centering, cfg=cfg, n_components=components)
        embedded = kernel_matrix @ projection

    embedded = _standardize_embedding(embedded)
    source_latent = embedded[:n_source].astype(np.float32, copy=False)
    target_latent = embedded[n_source:].astype(np.float32, copy=False)
    metadata = _metadata(
        cfg=cfg,
        n_source=n_source,
        n_target=n_target,
        original_dim=source.shape[1],
        latent_dim=source_latent.shape[1],
        eigenvalues=eigenvalues,
    )
    return TransferComponentResult(
        source_features=source_latent,
        target_features=target_latent,
        config=cfg,
        projection=projection.astype(np.float32, copy=False),
        eigenvalues=tuple(float(value) for value in eigenvalues),
        source_mean=mean.astype(np.float32, copy=False),
        source_scale=scale.astype(np.float32, copy=False),
        metadata=metadata,
    )


# pylint: disable-next=too-many-arguments

def fit_transfer_component_classifier(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    config: TransferComponentConfig | Mapping[str, Any] | None = None,
    classifier: BaseEstimator | None = None,
    classifier_C: float | str = 1.0,
    classifier_max_iter: int | str = 1000,
    classifier_class_weight: str | Mapping[Any, float] | None = "balanced",
    sample_weight: Sequence[float] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
) -> TransferComponentClassificationResult:
    """Fit TCA and a source-label classifier in the latent space."""

    if target_labels is not None:
        raise ValueError("TCA classification does not accept target_labels; target labels must be reserved for scoring.")
    labels = _label_vector(source_labels, expected_length=_feature_matrix(source_features, name="source_features").shape[0], name="source_labels")
    encoded_labels, label_classes = _encode_atomic_labels(labels)
    if label_classes.shape[0] < 2:
        raise ValueError("source_labels must contain at least two classes.")
    tca = fit_transfer_component_features(source_features=source_features, target_features=target_features, config=config)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights is not None:
        if weights.shape[0] != labels.shape[0]:
            raise ValueError(f"sample_weight must contain one value per source row: {weights.shape[0]} != {labels.shape[0]}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weight must contain finite non-negative values.")
    model = clone(classifier) if classifier is not None else LogisticRegression(
        C=_positive_float(classifier_C, name="classifier_C"),
        max_iter=_positive_int(classifier_max_iter, name="classifier_max_iter"),
        class_weight=classifier_class_weight,
        random_state=13,
    )
    fit_kwargs = {} if weights is None else {"sample_weight": weights}
    model.fit(tca.source_features, encoded_labels, **fit_kwargs)
    predictions = _decode_label_codes(np.asarray(model.predict(tca.target_features)), label_classes)
    probabilities = _predict_probabilities_or_none(model, tca.target_features)
    encoded_classes = np.asarray(getattr(model, "classes_", np.arange(label_classes.shape[0])), dtype=int)
    classes = _decode_label_codes(encoded_classes, label_classes)
    metadata = {
        **tca.metadata,
        "transfer_component_classifier": type(model).__name__,
        "transfer_component_classifier_uses_source_labels": True,
        "transfer_component_classifier_uses_target_labels": False,
        "transfer_component_classifier_label_encoding": "atomic_integer",
    }
    return TransferComponentClassificationResult(
        source_features=tca.source_features,
        target_features=tca.target_features,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
        classifier=model,
        tca=tca,
        metadata=metadata,
    )


def normalize_transfer_component_kernel(value: str | None) -> str:
    """Normalize TCA kernel aliases."""

    normalized = "linear" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"lin": "linear", "identity": "linear", "gaussian": "rbf", "rbf_kernel": "rbf"}.get(normalized, normalized)
    if normalized not in TRANSFER_COMPONENT_KERNELS:
        raise ValueError(f"Unknown TCA kernel {value!r}. Available kernels: {', '.join(TRANSFER_COMPONENT_KERNELS)}.")
    return normalized


def normalize_standardize_scope(value: str | None) -> str:
    """Normalize standardization-scope aliases."""

    normalized = "source" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "off": "none",
        "false": "none",
        "raw": "none",
        "source_only": "source",
        "train": "source",
        "training": "source",
        "all": "source_target",
        "source+target": "source_target",
        "source_and_target": "source_target",
        "unlabeled_target": "source_target",
    }.get(normalized, normalized)
    if normalized not in TRANSFER_COMPONENT_STANDARDIZE_SCOPES:
        raise ValueError(f"Unknown standardize_scope {value!r}. Available scopes: {', '.join(TRANSFER_COMPONENT_STANDARDIZE_SCOPES)}.")
    return normalized


def _coerce_config(config: TransferComponentConfig | Mapping[str, Any] | None) -> TransferComponentConfig:
    if config is None:
        return transfer_component_config()
    if isinstance(config, TransferComponentConfig):
        return config
    return transfer_component_config(**dict(config))


def _fit_linear_tca(combined: np.ndarray, *, mmd: np.ndarray, centering: np.ndarray, cfg: TransferComponentConfig, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    left = combined.T @ mmd @ combined + cfg.regularization * np.eye(combined.shape[1], dtype=float)
    right = combined.T @ centering @ combined + cfg.epsilon * np.eye(combined.shape[1], dtype=float)
    return _solve_generalized(left, right, n_components=n_components)


def _fit_kernel_tca(kernel_matrix: np.ndarray, *, mmd: np.ndarray, centering: np.ndarray, cfg: TransferComponentConfig, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    left = kernel_matrix @ mmd @ kernel_matrix + cfg.regularization * np.eye(kernel_matrix.shape[0], dtype=float)
    right = kernel_matrix @ centering @ kernel_matrix + cfg.epsilon * np.eye(kernel_matrix.shape[0], dtype=float)
    return _solve_generalized(left, right, n_components=n_components)


def _solve_generalized(left: np.ndarray, right: np.ndarray, *, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    left = _symmetrize(left)
    right = _symmetrize(right)
    try:
        eigenvalues, eigenvectors = eigh(left, right, check_finite=True)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * max(1.0, float(np.trace(right) / max(1, right.shape[0])))
        eigenvalues, eigenvectors = eigh(left, right + jitter * np.eye(right.shape[0]), check_finite=True)
    order = np.argsort(eigenvalues)
    selected = order[:n_components]
    projection = eigenvectors[:, selected]
    norms = np.linalg.norm(projection, axis=0, keepdims=True)
    projection = projection / np.maximum(norms, 1e-12)
    return projection, np.asarray(eigenvalues[selected], dtype=float)


def _mmd_matrix(n_source: int, n_target: int) -> np.ndarray:
    if n_source < 1 or n_target < 1:
        raise ValueError("TCA requires at least one source row and one target row.")
    weights = np.concatenate([np.full(n_source, 1.0 / n_source), np.full(n_target, -1.0 / n_target)])
    matrix = np.outer(weights, weights)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    return matrix / max(norm, 1e-12)


def _kernel_matrix(features: np.ndarray, *, kernel: str, gamma: float | str | None) -> np.ndarray:
    if kernel == "linear":
        return features @ features.T
    if kernel == "rbf":
        squared = _squared_euclidean(features, features)
        resolved_gamma = _resolve_gamma(gamma, features=features, squared_distances=squared)
        return np.exp(-resolved_gamma * squared)
    raise ValueError(f"Unhandled kernel {kernel!r}.")


def _resolve_gamma(gamma: float | str | None, *, features: np.ndarray, squared_distances: np.ndarray) -> float:
    if gamma is None:
        gamma = "scale"
    if isinstance(gamma, str):
        normalized = gamma.strip().lower().replace("-", "_")
        if normalized in {"scale", "auto"}:
            variance = float(np.var(features))
            return 1.0 / max(features.shape[1] * variance, 1e-12)
        if normalized in {"median", "median_distance"}:
            positive = squared_distances[squared_distances > 0.0]
            return 1.0 / max(float(np.median(positive)), 1e-12) if positive.size else 1.0
        gamma = float(normalized)
    value = float(gamma)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("gamma must be positive, 'scale', 'auto', or 'median'.")
    return value


def _standardize_source_target(source: np.ndarray, target: np.ndarray, *, scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scope == "none":
        mean = np.zeros(source.shape[1], dtype=float)
        scale = np.ones(source.shape[1], dtype=float)
    elif scope == "source":
        mean = np.mean(source, axis=0)
        scale = np.std(source, axis=0, ddof=1 if source.shape[0] > 1 else 0)
    elif scope == "source_target":
        combined = np.vstack([source, target])
        mean = np.mean(combined, axis=0)
        scale = np.std(combined, axis=0, ddof=1 if combined.shape[0] > 1 else 0)
    else:  # pragma: no cover - normalized before use
        raise ValueError(f"Unhandled standardize_scope {scope!r}.")
    scale = np.maximum(scale, 1e-12)
    return (source - mean) / scale, (target - mean) / scale, mean, scale


def _standardize_embedding(embedding: np.ndarray) -> np.ndarray:
    mean = np.mean(embedding, axis=0)
    scale = np.std(embedding, axis=0, ddof=1 if embedding.shape[0] > 1 else 0)
    return (embedding - mean) / np.maximum(scale, 1e-12)


def _metadata(*, cfg: TransferComponentConfig, n_source: int, n_target: int, original_dim: int, latent_dim: int, eigenvalues: np.ndarray) -> dict[str, Any]:
    return {
        "transfer_component_analysis": True,
        "transfer_component_protocol": TRANSFER_COMPONENT_PROTOCOL,
        "transfer_component_protocol_category": TRANSFER_COMPONENT_CATEGORY,
        "transfer_component_uses_source_features": True,
        "transfer_component_uses_source_labels": False,
        "transfer_component_uses_target_features": True,
        "transfer_component_uses_target_labels": False,
        "transfer_component_valid_for_strict_source_only": False,
        "transfer_component_valid_for_unlabeled_target_adaptation": True,
        "transfer_component_valid_for_target_calibration": False,
        "transfer_component_kernel": cfg.kernel,
        "transfer_component_n_source_rows": int(n_source),
        "transfer_component_n_target_rows": int(n_target),
        "transfer_component_original_dim": int(original_dim),
        "transfer_component_latent_dim": int(latent_dim),
        "transfer_component_regularization": float(cfg.regularization),
        "transfer_component_gamma": "" if cfg.gamma is None else str(cfg.gamma),
        "transfer_component_standardize_scope": cfg.standardize_scope,
        "transfer_component_eigenvalues": "|".join(f"{float(value):.12g}" for value in eigenvalues),
    }


def _predict_probabilities_or_none(model: BaseEstimator, features: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return _normalize_probability_rows(np.asarray(model.predict_proba(features), dtype=float))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        return _normalize_probability_rows(np.exp(np.clip(shifted, -50.0, 50.0)))
    return None


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.maximum(np.asarray(probabilities, dtype=float), 0.0)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Predicted probability rows must have positive mass.")
    return matrix / row_sums


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional matrix with at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype == object and values.ndim == 1:
        items = [_atomic_label_value(value) for value in values.reshape(-1).tolist()]
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            raise ValueError(f"{name} must contain one value per source row.")
        if array.ndim == 1:
            items = [_atomic_label_value(value) for value in array.reshape(-1).tolist()]
        elif array.ndim == 2 and array.shape[1] == 1:
            items = [_atomic_label_value(value) for value in array[:, 0].tolist()]
        elif array.ndim == 2 and array.shape[0] == 1 and array.shape[1] == expected_length:
            items = [_atomic_label_value(value) for value in array.reshape(-1).tolist()]
        elif array.ndim == 2:
            items = [_atomic_label_value(tuple(row.tolist())) for row in array]
        else:
            raise ValueError(f"{name} must be a one-dimensional label vector or a row-wise composite-label matrix.")
    vector = _object_vector(items)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _atomic_label_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _atomic_label_value(value.item())
        return tuple(_atomic_label_value(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_atomic_label_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_atomic_label_value(item) for item in value)
    return value


def _object_vector(items: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _encode_atomic_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key_to_index: dict[Any, int] = {}
    classes: list[Any] = []
    encoded = np.empty(labels.shape[0], dtype=int)
    for index, label in enumerate(labels):
        key = _label_key(label)
        if key not in key_to_index:
            key_to_index[key] = len(classes)
            classes.append(label)
        encoded[index] = key_to_index[key]
    return encoded, _object_vector(classes)


def _decode_label_codes(codes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    code_array = np.asarray(codes)
    decoded = np.empty(code_array.size, dtype=object)
    for index, code in enumerate(code_array.reshape(-1)):
        class_index = _integer_code(code)
        if class_index < 0 or class_index >= classes.shape[0]:
            raise ValueError(f"Classifier returned an unknown encoded class {code!r}.")
        decoded[index] = classes[class_index]
    return decoded.reshape(code_array.shape)


def _integer_code(value: Any) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"Classifier returned a non-integer encoded class {value!r}.")
    return int(numeric)


def _label_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _label_key(value.item())
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(_label_key(item) for item in value.tolist()))
    if isinstance(value, Mapping):
        return ("mapping", tuple(sorted((_label_key(key), _label_key(item)) for key, item in value.items())))
    if isinstance(value, list):
        return ("list", tuple(_label_key(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_label_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return ("scalar", value)


def _normalize_components(value: int | str) -> int | str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full", "inf", "infinity"}:
            return "all"
        value = text
    return _positive_int(value, name="n_components")


def _effective_components(value: int | str, *, max_components: int) -> int:
    if max_components < 1:
        raise ValueError("At least one transfer component is required; provide more rows or features.")
    return max_components if value == "all" else min(int(value), int(max_components))


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed
