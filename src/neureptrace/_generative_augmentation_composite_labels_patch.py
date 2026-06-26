"""Preserve composite labels in feature-space generative augmentation."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from typing import Any

import numpy as np

_INSTALLED = False
_PATCH_MARKER = "_neureptrace_generative_augmentation_composite_labels_patch_installed"


def _contains_composite_label(items: Sequence[object]) -> bool:
    return any(_is_composite_label(item) for item in items)


def _is_composite_label(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim != 0
    return isinstance(value, (tuple, list, dict))


def _object_vector(items: Sequence[object]) -> np.ndarray:
    vector = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        vector[index] = item
    return vector


def _label_vector(labels: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            items = [labels.item()]
        elif labels.ndim == 1:
            items = list(labels)
        elif labels.shape[0] == expected_length and labels.shape[1] > 1:
            items = [tuple(row.tolist()) for row in labels.reshape(labels.shape[0], -1)]
        else:
            items = list(labels.reshape(-1))
    elif isinstance(labels, (str, bytes)):
        items = [labels]
    else:
        try:
            items = list(labels)
        except TypeError:
            items = [labels]

    if len(items) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(items)} != {expected_length}.")
    if _contains_composite_label(items):
        return _object_vector(items)
    return np.asarray(items).ravel()


def _as_python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _labels_equal(left: object, right: object) -> bool:
    left = _as_python_scalar(left)
    right = _as_python_scalar(right)
    try:
        comparison = left == right
    except (TypeError, ValueError):
        comparison = False
    if isinstance(comparison, np.ndarray):
        try:
            return bool(np.all(comparison))
        except (TypeError, ValueError):
            return False
    try:
        if bool(comparison):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.isscalar(left) and np.isscalar(right) and np.isnan(left) and np.isnan(right))
    except (TypeError, ValueError):
        return False


def _label_mask(labels: np.ndarray, class_label: object) -> np.ndarray:
    return np.asarray([_labels_equal(label, class_label) for label in labels], dtype=bool)


def _unique_labels(labels: np.ndarray) -> np.ndarray:
    unique: list[object] = []
    for label in labels:
        if not any(_labels_equal(label, existing) for existing in unique):
            unique.append(label)
    if _contains_composite_label(unique):
        return _object_vector(unique)
    return np.asarray(unique, dtype=labels.dtype)


def _unknown_labels(candidate_labels: np.ndarray, reference_labels: np.ndarray) -> list[object]:
    return [label for label in _unique_labels(candidate_labels) if not np.any(_label_mask(reference_labels, label))]


def _label_block(class_label: object, n_rows: int, *, label_dtype: np.dtype) -> np.ndarray:
    if _is_composite_label(class_label):
        return _object_vector([class_label] * int(n_rows))
    return np.full(int(n_rows), class_label, dtype=label_dtype)


def _generate_gaussian_rows(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    calibration_features: np.ndarray | None,
    calibration_labels: np.ndarray | None,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    module = importlib.import_module("neureptrace.decoding.generative_augmentation")
    rng = np.random.default_rng(config.random_state)
    synthetic_blocks: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []
    for class_label in _unique_labels(labels):
        class_rows = features[_label_mask(labels, class_label)]
        if class_rows.shape[0] == 0:
            continue
        calibration_rows = None
        if calibration_features is not None and calibration_labels is not None:
            calibration_rows = calibration_features[_label_mask(calibration_labels, class_label)]
        samples = module._sample_class_rows(
            class_rows,
            fallback_rows=features,
            calibration_rows=calibration_rows,
            config=config,
            rng=rng,
        )
        synthetic_blocks.append(samples)
        synthetic_labels.append(_label_block(class_label, samples.shape[0], label_dtype=labels.dtype))
    return module._stack_synthetic_blocks(synthetic_blocks, synthetic_labels, n_features=features.shape[1], label_dtype=labels.dtype)


def _apply_target_calibration_shift(
    synthetic_features: np.ndarray,
    synthetic_labels: np.ndarray,
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    weight: float,
) -> np.ndarray:
    if weight <= 0.0:
        return synthetic_features
    shifted = synthetic_features.copy()
    for class_label in _unique_labels(synthetic_labels):
        class_mask = _label_mask(synthetic_labels, class_label)
        source_rows = source_features[_label_mask(source_labels, class_label)]
        calibration_rows = calibration_features[_label_mask(calibration_labels, class_label)]
        if source_rows.shape[0] == 0 or calibration_rows.shape[0] == 0:
            continue
        delta = np.mean(calibration_rows, axis=0) - np.mean(source_rows, axis=0)
        shifted[class_mask] = shifted[class_mask] + weight * delta
    return shifted


def _label_index(class_labels: np.ndarray, label: object) -> int:
    for index, class_label in enumerate(class_labels):
        if _labels_equal(label, class_label):
            return int(index)
    raise KeyError(label)


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_labels = _unique_labels(labels)
    encoded = np.asarray([_label_index(class_labels, label) for label in labels], dtype=int)
    return encoded, class_labels


def _augment_training_features(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    *,
    config: Any = None,
    target_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
):
    module = importlib.import_module("neureptrace.decoding.generative_augmentation")
    cfg = module._coerce_config(config)
    features = module._feature_matrix(train_features, name="train_features")
    labels = _label_vector(train_labels, expected_length=features.shape[0], name="train_labels")
    if target_labels is not None:
        raise ValueError("Generative augmentation never accepts scored target_labels; use disjoint target_calibration_labels for category-3 calibration.")
    if not cfg.enabled:
        synthetic_mask = np.zeros(features.shape[0], dtype=bool)
        return module.GenerativeAugmentationResult(features=features, labels=labels, synthetic_mask=synthetic_mask, metadata=module._metadata(cfg, features.shape[0], 0))

    target_matrix: np.ndarray | None = None
    if cfg.method in module.TARGET_STYLE_AUGMENTATIONS:
        if target_features is None:
            raise ValueError(f"{cfg.method} requires unlabeled target_features.")
        target_matrix = module._feature_matrix(target_features, name="target_features")
        if target_matrix.shape[1] != features.shape[1]:
            raise ValueError("target_features must have the same feature dimension as train_features.")

    calibration_features: np.ndarray | None = None
    calibration_labels: np.ndarray | None = None
    if cfg.method in module.TARGET_CALIBRATED_AUGMENTATIONS:
        if target_calibration_features is None or target_calibration_labels is None:
            raise ValueError(f"{cfg.method} requires disjoint target_calibration_features and target_calibration_labels.")
        calibration_features = module._feature_matrix(target_calibration_features, name="target_calibration_features")
        if calibration_features.shape[1] != features.shape[1]:
            raise ValueError("target_calibration_features must have the same feature dimension as train_features.")
        calibration_labels = _label_vector(
            target_calibration_labels,
            expected_length=calibration_features.shape[0],
            name="target_calibration_labels",
        )
        if _unknown_labels(calibration_labels, labels):
            raise ValueError("target_calibration_labels must be a subset of train_labels.")

    if cfg.method in module.GAN_AUGMENTATIONS:
        synthetic_features, synthetic_labels = module._generate_neural_rows(
            features,
            labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            config=cfg,
            generator_kind="gan",
        )
    elif cfg.method in module.DIFFUSION_AUGMENTATIONS:
        synthetic_features, synthetic_labels = module._generate_neural_rows(
            features,
            labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            config=cfg,
            generator_kind="diffusion",
        )
    else:
        synthetic_features, synthetic_labels = _generate_gaussian_rows(
            features,
            labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            config=cfg,
        )

    if synthetic_features.shape[0] == 0:
        synthetic_mask = np.zeros(features.shape[0], dtype=bool)
        return module.GenerativeAugmentationResult(features=features, labels=labels, synthetic_mask=synthetic_mask, metadata=module._metadata(cfg, features.shape[0], 0))

    if cfg.method in module.TARGET_STYLE_AUGMENTATIONS and target_matrix is not None:
        synthetic_features = module._match_target_style(
            synthetic_features,
            source_features=features,
            target_features=target_matrix,
            strength=cfg.target_style_strength,
            floor=cfg.covariance_floor,
        )
    if cfg.method in module.TARGET_CALIBRATED_AUGMENTATIONS and calibration_features is not None and calibration_labels is not None and cfg.method in module.NEURAL_AUGMENTATIONS:
        synthetic_features = _apply_target_calibration_shift(
            synthetic_features,
            synthetic_labels,
            source_features=features,
            source_labels=labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            weight=cfg.target_calibration_weight,
        )

    new_features = np.vstack([features, synthetic_features])
    new_labels = np.concatenate([labels, synthetic_labels])
    synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
    return module.GenerativeAugmentationResult(
        features=new_features,
        labels=new_labels,
        synthetic_mask=synthetic_mask,
        metadata=module._metadata(cfg, features.shape[0], synthetic_features.shape[0]),
    )


def install() -> None:
    """Patch generative augmentation to keep composite class labels atomic."""

    global _INSTALLED
    if _INSTALLED:
        return

    module = importlib.import_module("neureptrace.decoding.generative_augmentation")
    if getattr(module.augment_training_features, _PATCH_MARKER, False):
        _INSTALLED = True
        return

    module._label_vector = _label_vector
    module._generate_gaussian_rows = _generate_gaussian_rows
    module._apply_target_calibration_shift = _apply_target_calibration_shift
    module._encode_labels = _encode_labels
    module.augment_training_features = _augment_training_features
    setattr(module.augment_training_features, _PATCH_MARKER, True)
    _INSTALLED = True


__all__ = ["install"]
