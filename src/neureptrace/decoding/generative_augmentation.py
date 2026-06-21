"""Feature-space generative augmentation for source-only and target-adaptive decoding.

The utilities in this module deliberately operate on already extracted feature
matrices.  They are lightweight enough for fold-local LOSO experiments and avoid
introducing a hard torch dependency for augmentation baselines.  They implement
three protocol-explicit variants:

* ``source_gaussian``: class-conditional Gaussian synthesis from source training
  rows only.  This is strict source-only augmentation.
* ``target_style_gaussian``: source class-conditional synthesis followed by
  unlabeled target mean/covariance style matching.  This is category-2
  target-adaptive augmentation and must be reported separately from strict
  source-only results.
* ``target_calibrated_gaussian``: few-shot target calibration rows/labels may
  shift class means before synthesis.  This is category-3 calibrated
  augmentation and must not be used as a zero-calibration benchmark.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

GEN_AUGMENTATION_METHODS = (
    "none",
    "source_gaussian",
    "target_style_gaussian",
    "target_calibrated_gaussian",
)
SOURCE_GAUSSIAN_AUGMENTATION = "source_gaussian"
TARGET_STYLE_GAUSSIAN_AUGMENTATION = "target_style_gaussian"
TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION = "target_calibrated_gaussian"

SOURCE_ONLY_GENERATIVE_PROTOCOL = "strict_source_only_synthetic_augmentation"
UNLABELED_TARGET_GENERATIVE_PROTOCOL = "unlabeled_target_generative_augmentation"
TARGET_CALIBRATED_GENERATIVE_PROTOCOL = "target_calibrated_generative_augmentation"


@dataclass(frozen=True, slots=True)
class GenerativeAugmentationConfig:
    """Configuration for fold-local feature-space generative augmentation."""

    method: str = "none"
    synthetic_per_class: int = 0
    noise_scale: float = 1.0
    covariance_shrinkage: float = 0.1
    covariance_floor: float = 1e-6
    random_state: int | None = 13
    target_style_strength: float = 1.0
    target_calibration_weight: float = 0.5

    @property
    def enabled(self) -> bool:
        return self.method != "none" and self.synthetic_per_class > 0

    @property
    def uses_unlabeled_target_data(self) -> bool:
        return self.enabled and self.method == TARGET_STYLE_GAUSSIAN_AUGMENTATION

    @property
    def target_calibrated(self) -> bool:
        return self.enabled and self.method == TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION

    @property
    def protocol(self) -> str:
        if self.target_calibrated:
            return TARGET_CALIBRATED_GENERATIVE_PROTOCOL
        if self.uses_unlabeled_target_data:
            return UNLABELED_TARGET_GENERATIVE_PROTOCOL
        return SOURCE_ONLY_GENERATIVE_PROTOCOL

    @property
    def protocol_category(self) -> int:
        if self.target_calibrated:
            return 3
        if self.uses_unlabeled_target_data:
            return 2
        return 1


@dataclass(frozen=True, slots=True)
class GenerativeAugmentationResult:
    """Augmented features plus provenance metadata."""

    features: np.ndarray
    labels: np.ndarray
    synthetic_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_synthetic(self) -> int:
        return int(np.sum(self.synthetic_mask))


def generative_augmentation_config(
    *,
    method: str | None = None,
    synthetic_per_class: int | str = 0,
    noise_scale: float | str = 1.0,
    covariance_shrinkage: float | str = 0.1,
    covariance_floor: float | str = 1e-6,
    random_state: int | str | None = 13,
    target_style_strength: float | str = 1.0,
    target_calibration_weight: float | str = 0.5,
) -> GenerativeAugmentationConfig:
    """Normalize user-facing generative-augmentation options."""

    return GenerativeAugmentationConfig(
        method=normalize_generative_augmentation_method(method),
        synthetic_per_class=_normalize_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
        noise_scale=_normalize_nonnegative_float(noise_scale, name="noise_scale"),
        covariance_shrinkage=_normalize_unit_interval(covariance_shrinkage, name="covariance_shrinkage"),
        covariance_floor=_normalize_nonnegative_float(covariance_floor, name="covariance_floor"),
        random_state=None if random_state in {None, "", "none", "None"} else _normalize_integer(random_state, name="random_state"),
        target_style_strength=_normalize_unit_interval(target_style_strength, name="target_style_strength"),
        target_calibration_weight=_normalize_unit_interval(target_calibration_weight, name="target_calibration_weight"),
    )


def normalize_generative_augmentation_method(method: str | None) -> str:
    """Normalize aliases for feature-space generative augmentation."""

    normalized = "none" if method is None else str(method).strip().lower().replace("-", "_")
    if normalized in {"", "off", "false", "identity", "raw", "no", "disabled"}:
        normalized = "none"
    normalized = {
        "gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "class_gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "source_class_gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "source_only_gaussian": SOURCE_GAUSSIAN_AUGMENTATION,
        "target_style": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "target_matched_gaussian": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "unlabeled_target_gaussian": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "style_transfer_gaussian": TARGET_STYLE_GAUSSIAN_AUGMENTATION,
        "few_shot_gaussian": TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
        "calibrated_gaussian": TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
        "target_calibrated": TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION,
    }.get(normalized, normalized)
    if normalized not in GEN_AUGMENTATION_METHODS:
        raise ValueError(
            f"Unknown generative augmentation method {method!r}. "
            f"Available methods: {', '.join(GEN_AUGMENTATION_METHODS)}."
        )
    return normalized


def augment_training_features(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    *,
    config: GenerativeAugmentationConfig | Mapping[str, Any] | None = None,
    target_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
) -> GenerativeAugmentationResult:
    """Append synthetic feature rows generated within an explicit protocol.

    ``target_labels`` are intentionally rejected.  Category-3 calibration must be
    supplied through ``target_calibration_features`` and
    ``target_calibration_labels`` so scored target labels cannot be passed
    accidentally.
    """

    cfg = _coerce_config(config)
    features = _feature_matrix(train_features, name="train_features")
    labels = _label_vector(train_labels, expected_length=features.shape[0], name="train_labels")
    if target_labels is not None:
        raise ValueError("Generative augmentation never accepts scored target_labels; use disjoint target_calibration_labels for category-3 calibration.")
    if not cfg.enabled:
        synthetic_mask = np.zeros(features.shape[0], dtype=bool)
        return GenerativeAugmentationResult(features=features, labels=labels, synthetic_mask=synthetic_mask, metadata=_metadata(cfg, features.shape[0], 0))

    target_matrix: np.ndarray | None = None
    if cfg.method == TARGET_STYLE_GAUSSIAN_AUGMENTATION:
        if target_features is None:
            raise ValueError("target_style_gaussian requires unlabeled target_features.")
        target_matrix = _feature_matrix(target_features, name="target_features")
        if target_matrix.shape[1] != features.shape[1]:
            raise ValueError("target_features must have the same feature dimension as train_features.")

    calibration_features: np.ndarray | None = None
    calibration_labels: np.ndarray | None = None
    if cfg.method == TARGET_CALIBRATED_GAUSSIAN_AUGMENTATION:
        if target_calibration_features is None or target_calibration_labels is None:
            raise ValueError("target_calibrated_gaussian requires disjoint target_calibration_features and target_calibration_labels.")
        calibration_features = _feature_matrix(target_calibration_features, name="target_calibration_features")
        if calibration_features.shape[1] != features.shape[1]:
            raise ValueError("target_calibration_features must have the same feature dimension as train_features.")
        calibration_labels = _label_vector(
            target_calibration_labels,
            expected_length=calibration_features.shape[0],
            name="target_calibration_labels",
        )

    rng = np.random.default_rng(cfg.random_state)
    synthetic_blocks: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []
    for class_label in np.unique(labels):
        class_rows = features[labels == class_label]
        if class_rows.shape[0] == 0:
            continue
        calibration_rows = None
        if calibration_features is not None and calibration_labels is not None:
            calibration_rows = calibration_features[calibration_labels == class_label]
        samples = _sample_class_rows(
            class_rows,
            fallback_rows=features,
            calibration_rows=calibration_rows,
            config=cfg,
            rng=rng,
        )
        synthetic_blocks.append(samples)
        synthetic_labels.append(np.full(samples.shape[0], class_label, dtype=labels.dtype))

    if not synthetic_blocks:
        synthetic_mask = np.zeros(features.shape[0], dtype=bool)
        return GenerativeAugmentationResult(features=features, labels=labels, synthetic_mask=synthetic_mask, metadata=_metadata(cfg, features.shape[0], 0))

    synthetic_features = np.vstack(synthetic_blocks)
    if cfg.method == TARGET_STYLE_GAUSSIAN_AUGMENTATION and target_matrix is not None:
        synthetic_features = _match_target_style(
            synthetic_features,
            source_features=features,
            target_features=target_matrix,
            strength=cfg.target_style_strength,
            floor=cfg.covariance_floor,
        )
    new_features = np.vstack([features, synthetic_features])
    new_labels = np.concatenate([labels, *synthetic_labels])
    synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    return GenerativeAugmentationResult(
        features=new_features,
        labels=new_labels,
        synthetic_mask=synthetic_mask,
        metadata=_metadata(cfg, features.shape[0], synthetic_features.shape[0]),
    )


def make_generative_augmented_fit_model(
    fit_model: Callable[[np.ndarray, np.ndarray], Any],
    *,
    config: GenerativeAugmentationConfig | Mapping[str, Any] | None = None,
    target_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray], Any]:
    """Return a ``fit_model`` wrapper that augments only the current training fold."""

    cfg = _coerce_config(config)

    def _fit(features: np.ndarray, labels: np.ndarray):
        augmented = augment_training_features(
            features,
            labels,
            config=cfg,
            target_features=target_features,
            target_calibration_features=target_calibration_features,
            target_calibration_labels=target_calibration_labels,
        )
        model = fit_model(augmented.features, augmented.labels)
        try:
            setattr(model, "generative_augmentation_metadata_", augmented.metadata)
        except Exception:  # pragma: no cover - some sklearn wrappers disallow dynamic attrs
            pass
        return model

    return _fit


def _sample_class_rows(
    class_rows: np.ndarray,
    *,
    fallback_rows: np.ndarray,
    calibration_rows: np.ndarray | None,
    config: GenerativeAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.mean(class_rows, axis=0)
    covariance_rows = class_rows
    if calibration_rows is not None and calibration_rows.shape[0] > 0:
        calibration_mean = np.mean(calibration_rows, axis=0)
        mean = (1.0 - config.target_calibration_weight) * mean + config.target_calibration_weight * calibration_mean
        covariance_rows = np.vstack([class_rows, calibration_rows])
    covariance = _regularized_covariance(
        covariance_rows,
        fallback_rows=fallback_rows,
        shrinkage=config.covariance_shrinkage,
        floor=config.covariance_floor,
    )
    sqrt_covariance = _matrix_power(covariance, 0.5, floor=config.covariance_floor)
    noise = rng.normal(size=(config.synthetic_per_class, class_rows.shape[1]))
    return mean + config.noise_scale * (noise @ sqrt_covariance.T)


def _match_target_style(
    synthetic_features: np.ndarray,
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    strength: float,
    floor: float,
) -> np.ndarray:
    if strength <= 0.0:
        return synthetic_features
    source_mean = np.mean(source_features, axis=0)
    target_mean = np.mean(target_features, axis=0)
    source_covariance = _regularized_covariance(source_features, fallback_rows=source_features, shrinkage=0.0, floor=floor)
    target_covariance = _regularized_covariance(target_features, fallback_rows=target_features, shrinkage=0.0, floor=floor)
    source_inv_sqrt = _matrix_power(source_covariance, -0.5, floor=floor)
    target_sqrt = _matrix_power(target_covariance, 0.5, floor=floor)
    matched = target_mean + (synthetic_features - source_mean) @ source_inv_sqrt @ target_sqrt
    return (1.0 - strength) * synthetic_features + strength * matched


def _regularized_covariance(
    rows: np.ndarray,
    *,
    fallback_rows: np.ndarray,
    shrinkage: float,
    floor: float,
) -> np.ndarray:
    rows = _feature_matrix(rows, name="rows")
    fallback_rows = _feature_matrix(fallback_rows, name="fallback_rows")
    n_features = rows.shape[1]
    if rows.shape[0] > 1:
        covariance = np.cov(rows, rowvar=False)
    elif fallback_rows.shape[0] > 1:
        covariance = np.cov(fallback_rows, rowvar=False)
    else:
        covariance = np.eye(n_features, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    if not np.all(np.isfinite(covariance)):
        covariance = np.eye(n_features, dtype=float)
    return covariance + max(float(floor), 0.0) * np.eye(n_features, dtype=float)


def _matrix_power(matrix: np.ndarray, power: float, *, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    safe_values = np.maximum(values, max(float(floor), 0.0))
    return (vectors * np.power(safe_values, power)) @ vectors.T


def _coerce_config(config: GenerativeAugmentationConfig | Mapping[str, Any] | None) -> GenerativeAugmentationConfig:
    if config is None:
        return generative_augmentation_config()
    if isinstance(config, GenerativeAugmentationConfig):
        return config
    if isinstance(config, Mapping):
        return generative_augmentation_config(**dict(config))
    raise TypeError("config must be a GenerativeAugmentationConfig, a mapping, or None.")


def _metadata(config: GenerativeAugmentationConfig, n_real: int, n_synthetic: int) -> dict[str, Any]:
    return {
        "generative_augmentation_method": config.method,
        "generative_augmentation_enabled": bool(config.enabled),
        "generative_augmentation_protocol": config.protocol,
        "generative_augmentation_protocol_category": int(config.protocol_category),
        "generative_augmentation_synthetic_per_class": int(config.synthetic_per_class),
        "generative_augmentation_n_real": int(n_real),
        "generative_augmentation_n_synthetic": int(n_synthetic),
        "generative_augmentation_uses_unlabeled_target_data": bool(config.uses_unlabeled_target_data),
        "generative_augmentation_target_calibrated": bool(config.target_calibrated),
        "generative_augmentation_uses_target_labels": bool(config.target_calibrated),
        "generative_augmentation_valid_for_strict_source_only": bool(config.enabled and config.protocol_category == 1),
        "generative_augmentation_valid_for_benchmark": bool(config.protocol_category in {1, 2, 3}),
        "generative_augmentation_protocol_note": _protocol_note(config),
    }


def _protocol_note(config: GenerativeAugmentationConfig) -> str:
    if not config.enabled:
        return ""
    if config.method == SOURCE_GAUSSIAN_AUGMENTATION:
        return "source-only synthetic feature augmentation"
    if config.method == TARGET_STYLE_GAUSSIAN_AUGMENTATION:
        return "uses unlabeled target features for distribution/style matching; category-2 target-adaptive augmentation"
    return "uses disjoint labeled target calibration rows; category-3 supervised/calibrated augmentation"


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return matrix


def _label_vector(labels: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector


def _normalize_integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _normalize_nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _normalize_integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


def _normalize_nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return numeric


def _normalize_unit_interval(value: float | str, *, name: str) -> float:
    numeric = _normalize_nonnegative_float(value, name=name)
    if numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric
