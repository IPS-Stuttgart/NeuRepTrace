"""Runtime patch for source-decoder label vector normalization."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

from neureptrace._object_label_utils import values_equal

_DANN_ROW_ERROR = "DANN source_features and source_labels must contain the same rows."
_DANN_SHAPE_ERROR = "DANN source_labels must contain one label per source row."
_SOURCE_DG_ROW_ERROR = "source_features and source_labels must contain the same rows."
_SOURCE_DG_SHAPE_ERROR = "source-domain generalization source_labels must contain one label per source row."
_SOURCE_ROLL_LABEL_ROW_ERROR = "source_labels must contain one value per feature row."
_SOURCE_ROLL_LABEL_SHAPE_ERROR = "source_labels must contain one value per feature row."
_SOURCE_ROLL_DOMAIN_ROW_ERROR = "source_domains must contain one value per feature row."
_SOURCE_ROLL_DOMAIN_SHAPE_ERROR = "source_domains must contain one value per feature row."
_SOURCE_PROTOTYPE_CLASS_PATCH_MARKER = "_source_prototype_explicit_class_vector_patched"
_SOURCE_ROLL_COMPOSITE_IDENTIFIER_PATCH_MARKER = "_source_roll_composite_identifiers_patched"


def _source_row_count(source_features: Any) -> int | None:
    features = np.asarray(source_features)
    if features.ndim == 0:
        return None
    return int(features.shape[0])


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _as_label_vector(source_labels: Any, *, n_rows: int, row_error: str, shape_error: str) -> np.ndarray:
    labels = np.asarray(source_labels, dtype=object)
    if labels.ndim == 0 or labels.shape[0] != n_rows:
        raise ValueError(row_error)
    if labels.ndim == 1:
        return labels.reshape(n_rows)

    flattened_width = int(np.prod(labels.shape[1:], dtype=np.int64))
    if flattened_width == 1:
        return labels.reshape(n_rows)
    if flattened_width < 1:
        raise ValueError(shape_error)

    rows = labels.reshape(n_rows, flattened_width)
    return _object_value_vector(tuple(row.tolist()) for row in rows)


def _as_source_roll_label_vector(source_labels: Any, *, n_rows: int) -> np.ndarray:
    labels = _as_label_vector(
        source_labels,
        n_rows=n_rows,
        row_error=_SOURCE_ROLL_LABEL_ROW_ERROR,
        shape_error=_SOURCE_ROLL_LABEL_SHAPE_ERROR,
    )
    try:
        raw = np.asarray(source_labels)
    except ValueError:
        return labels
    if raw.ndim == 1 and raw.shape[0] == n_rows and raw.dtype != object:
        return raw.reshape(n_rows)
    return labels


def _as_source_roll_domain_vector(source_domains: Any, *, n_rows: int) -> np.ndarray:
    if source_domains is None:
        return np.full(n_rows, "source", dtype=object)
    domains = _as_label_vector(
        source_domains,
        n_rows=n_rows,
        row_error=_SOURCE_ROLL_DOMAIN_ROW_ERROR,
        shape_error=_SOURCE_ROLL_DOMAIN_SHAPE_ERROR,
    )
    for value in domains.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {value!r}.") from exc
    return domains


def _ordered_unique_values(values: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for value in values.tolist():
        if not any(values_equal(value, known) for known in unique):
            unique.append(value)
    return unique


def _value_mask(values: np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([values_equal(value, target) for value in values.tolist()], dtype=bool)


def _synthetic_label_vector(labels: Iterable[Any], dtype: np.dtype) -> np.ndarray:
    items = list(labels)
    if np.dtype(dtype).kind == "O":
        return _object_value_vector(items)
    return np.asarray(items, dtype=dtype)


def _as_class_vector(classes: Any) -> np.ndarray:
    """Return explicit class labels without flattening composite row labels."""

    if isinstance(classes, (str, bytes)):
        return _object_value_vector([classes])

    class_array = np.asarray(classes, dtype=object)
    if class_array.ndim == 0:
        return _object_value_vector([class_array.item()])
    if class_array.ndim == 1:
        return class_array.reshape(-1)
    if class_array.shape[0] == 0:
        return _object_value_vector([])

    flattened_width = int(np.prod(class_array.shape[1:], dtype=np.int64))
    if flattened_width == 1:
        return class_array.reshape(class_array.shape[0])

    rows = class_array.reshape(class_array.shape[0], flattened_width)
    return _object_value_vector(tuple(row.tolist()) for row in rows)


def _install_source_prototype_class_patch() -> None:
    source_prototypes = importlib.import_module("neureptrace.decoding.source_prototype_features")
    if getattr(source_prototypes.class_prototypes, _SOURCE_PROTOTYPE_CLASS_PATCH_MARKER, False):
        return

    original_class_prototypes = source_prototypes.class_prototypes

    @wraps(original_class_prototypes)
    def class_prototypes(source_features: Any, source_labels: Any, *, classes: Any = None) -> np.ndarray:
        if classes is not None:
            classes = _as_class_vector(classes)
        return original_class_prototypes(source_features, source_labels, classes=classes)

    setattr(class_prototypes, _SOURCE_PROTOTYPE_CLASS_PATCH_MARKER, True)
    source_prototypes.class_prototypes = class_prototypes


def _install_source_roll_composite_identifier_patch() -> None:
    source_roll = importlib.import_module("neureptrace.decoding.source_roll")
    if getattr(source_roll.augment_source_with_feature_roll, _SOURCE_ROLL_COMPOSITE_IDENTIFIER_PATCH_MARKER, False):
        return

    original_augment = source_roll.augment_source_with_feature_roll

    @wraps(original_augment)
    def augment_source_with_feature_roll(source_features: Any, source_labels: Any, *, source_domains: Any = None, config: Any = None):
        cfg = source_roll.source_feature_roll_config() if config is None else source_roll._coerce_config(config)
        features = source_roll._feature_matrix(source_features, name="source_features")
        labels = _as_source_roll_label_vector(source_labels, n_rows=features.shape[0])
        domains = _as_source_roll_domain_vector(source_domains, n_rows=features.shape[0])
        classes = _ordered_unique_values(labels)
        n_source_domains = len(_ordered_unique_values(domains))

        if not cfg.enabled:
            metadata = source_roll._metadata(
                cfg,
                n_source_rows=features.shape[0],
                n_synthetic_rows=0,
                n_classes=len(classes),
                n_source_domains=n_source_domains,
                feature_dim=features.shape[1],
            )
            return source_roll.SourceFeatureRollResult(
                features=features.astype(np.float32, copy=False),
                labels=labels.copy(),
                synthetic_mask=np.zeros(features.shape[0], dtype=bool),
                content_indices=np.empty(0, dtype=int),
                shifts=np.empty(0, dtype=int),
                metadata=metadata,
            )

        rng = np.random.default_rng(cfg.random_state)
        synthetic_rows: list[np.ndarray] = []
        synthetic_labels: list[Any] = []
        content_indices: list[int] = []
        shifts: list[int] = []
        for class_label in classes:
            class_indices = np.flatnonzero(_value_mask(labels, class_label))
            if class_indices.size == 0:
                continue
            for _ in range(cfg.synthetic_per_class):
                content_index = int(rng.choice(class_indices))
                shift = source_roll.sample_roll_shift(cfg.max_shift, include_zero_shift=cfg.include_zero_shift, rng=rng)
                synthetic_rows.append(
                    source_roll.roll_feature_row(
                        features[content_index],
                        shift=shift,
                        mode=cfg.roll_mode,
                        fill_value=cfg.fill_value,
                    )
                )
                synthetic_labels.append(class_label)
                content_indices.append(content_index)
                shifts.append(shift)

        synthetic_features = np.vstack(synthetic_rows).astype(np.float32, copy=False) if synthetic_rows else np.empty((0, features.shape[1]), dtype=np.float32)
        synthetic_labels_array = _synthetic_label_vector(synthetic_labels, labels.dtype)
        if cfg.preserve_original:
            output_features = np.vstack([features, synthetic_features]).astype(np.float32, copy=False)
            output_labels = np.concatenate([labels, synthetic_labels_array])
            synthetic_mask = np.concatenate([np.zeros(features.shape[0], dtype=bool), np.ones(synthetic_features.shape[0], dtype=bool)])
        else:
            output_features = synthetic_features
            output_labels = synthetic_labels_array
            synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

        metadata = source_roll._metadata(
            cfg,
            n_source_rows=features.shape[0],
            n_synthetic_rows=synthetic_features.shape[0],
            n_classes=len(classes),
            n_source_domains=n_source_domains,
            feature_dim=features.shape[1],
        )
        return source_roll.SourceFeatureRollResult(
            features=output_features,
            labels=output_labels,
            synthetic_mask=synthetic_mask,
            content_indices=np.asarray(content_indices, dtype=int),
            shifts=np.asarray(shifts, dtype=int),
            metadata=metadata,
        )

    setattr(augment_source_with_feature_roll, _SOURCE_ROLL_COMPOSITE_IDENTIFIER_PATCH_MARKER, True)
    augment_source_with_feature_roll.__wrapped__ = original_augment
    source_roll.augment_source_with_feature_roll = augment_source_with_feature_roll


def install() -> None:
    """Install source-label vector normalization for neural source decoders."""
    importlib.import_module("neureptrace._source_mixup_lambda_patch").install()
    source_centroid_patch = importlib.import_module("neureptrace._source_centroid_numeric_config_patch")
    source_centroid_patch.install()
    _install_source_prototype_class_patch()
    _install_source_roll_composite_identifier_patch()

    import neureptrace.decoding.dann as dann
    import neureptrace.decoding.source_domain_generalization as source_dg

    if not getattr(dann.TorchDANNClassifier.fit, "_source_label_vector_patched", False):
        original_dann_fit = dann.TorchDANNClassifier.fit

        @wraps(original_dann_fit)
        def fit_dann(self, source_features: np.ndarray, source_labels: np.ndarray, *, target_features: np.ndarray):
            n_rows = _source_row_count(source_features)
            if n_rows is not None:
                source_labels = _as_label_vector(source_labels, n_rows=n_rows, row_error=_DANN_ROW_ERROR, shape_error=_DANN_SHAPE_ERROR)
            return original_dann_fit(self, source_features, source_labels, target_features=target_features)

        fit_dann._source_label_vector_patched = True  # type: ignore[attr-defined]
        dann.TorchDANNClassifier.fit = fit_dann

    if not getattr(source_dg.TorchSourceDomainGeneralizationClassifier.fit, "_source_label_vector_patched", False):
        original_source_dg_fit = source_dg.TorchSourceDomainGeneralizationClassifier.fit

        @wraps(original_source_dg_fit)
        def fit_source_dg(self, source_features: np.ndarray, source_labels: np.ndarray, *, source_domains: np.ndarray):
            n_rows = _source_row_count(source_features)
            if n_rows is not None:
                source_labels = _as_label_vector(source_labels, n_rows=n_rows, row_error=_SOURCE_DG_ROW_ERROR, shape_error=_SOURCE_DG_SHAPE_ERROR)
            return original_source_dg_fit(self, source_features, source_labels, source_domains=source_domains)

        fit_source_dg._source_label_vector_patched = True  # type: ignore[attr-defined]
        source_dg.TorchSourceDomainGeneralizationClassifier.fit = fit_source_dg


__all__ = ["install"]
