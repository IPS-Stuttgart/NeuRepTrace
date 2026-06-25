"""Preserve composite labels in reconstruction-latent classifiers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import wraps
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

_PATCH_MARKER = "_neureptrace_reconstruction_tuple_labels_patch_installed"


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _atomic_label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    """Return one object-valued label per row without flattening composites."""

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        vector = _object_value_vector([array.item()])
    elif array.ndim == 1:
        if array.shape[0] == expected_length:
            vector = _object_value_vector(array.tolist())
        elif expected_length == 1:
            vector = _object_value_vector([tuple(array.tolist())])
        else:
            vector = _object_value_vector(array.reshape(-1).tolist())
    else:
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            vector = _object_value_vector(rows[:, 0].tolist())
        else:
            vector = _object_value_vector(tuple(row.tolist()) for row in rows)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
    return vector


def _values_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except Exception:  # pragma: no cover - defensive fallback for unusual metadata objects
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except Exception:  # pragma: no cover - defensive fallback for unusual metadata objects
        return False


def _label_codes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return atomic unique labels and integer codes preserving first-seen order."""

    unique: list[Any] = []
    codes: list[int] = []
    for label in labels.tolist():
        for code, candidate in enumerate(unique):
            if _values_equal(label, candidate):
                codes.append(code)
                break
        else:
            unique.append(label)
            codes.append(len(unique) - 1)
    return _object_value_vector(unique), np.asarray(codes, dtype=int)


def _requires_code_labels(labels: np.ndarray) -> bool:
    """Whether sklearn would treat row-level labels as sequences rather than classes."""

    return any(isinstance(label, (list, tuple, np.ndarray, dict)) for label in labels.tolist())


def _decode_label_codes(codes: Sequence[Any] | np.ndarray, classes: np.ndarray) -> np.ndarray:
    code_vector = np.asarray(codes, dtype=int).reshape(-1)
    return _object_value_vector(classes[int(code)] for code in code_vector)


def install() -> None:
    """Patch reconstruction-latent classifier labels so composite values remain atomic."""

    reconstruction_encoder = importlib.import_module("neureptrace.decoding.reconstruction_encoder")
    original = reconstruction_encoder.fit_reconstruction_latent_classifier
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def fit_reconstruction_latent_classifier(
        *,
        train_features: Sequence[Sequence[float]] | np.ndarray,
        train_labels: Sequence[Any] | np.ndarray,
        test_features: Sequence[Sequence[float]] | np.ndarray,
        config: Any = None,
        target_encoder_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        target_labels: Sequence[Any] | np.ndarray | None = None,
        classifier: BaseEstimator | None = None,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ):
        if target_labels is not None:
            raise ValueError("Reconstruction latent classifier does not accept target labels.")

        cfg = reconstruction_encoder.reconstruction_encoder_config() if config is None else config
        train_matrix = reconstruction_encoder._feature_matrix(train_features, name="train_features")
        labels = _atomic_label_vector(train_labels, expected_length=train_matrix.shape[0], name="train_labels")
        classes, label_codes = _label_codes(labels)
        if classes.shape[0] < 2:
            raise ValueError("train_labels must contain at least two classes.")

        latent = reconstruction_encoder.fit_reconstruction_latent_space(
            train_features=train_features,
            test_features=test_features,
            config=cfg,
            target_encoder_features=target_encoder_features,
        )

        encode_labels = _requires_code_labels(labels)
        fit_labels = label_codes if encode_labels else labels
        model = clone(classifier) if classifier is not None else LogisticRegression(
            C=cfg.classifier_C,
            class_weight=cfg.classifier_class_weight,
            max_iter=cfg.classifier_max_iter,
            random_state=cfg.random_state,
        )
        fit_kwargs = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight, dtype=float)}
        model.fit(latent.train_latent, fit_labels, **fit_kwargs)

        if encode_labels:
            predictions = _decode_label_codes(model.predict(latent.test_latent), classes)
            model_classes = np.asarray(getattr(model, "classes_", np.arange(classes.shape[0])), dtype=int)
            output_classes = _decode_label_codes(model_classes, classes)
            label_encoding = "integer_codes_for_composite_labels"
        else:
            predictions = np.asarray(model.predict(latent.test_latent))
            output_classes = np.asarray(getattr(model, "classes_", classes))
            label_encoding = "native"

        probabilities = np.asarray(model.predict_proba(latent.test_latent), dtype=float) if hasattr(model, "predict_proba") else None
        metadata = {
            **latent.metadata,
            "classifier_label_source": "source_train_labels",
            "classifier_target_labels_used": False,
            "classifier_name": type(model).__name__,
            "classifier_n_classes": int(output_classes.shape[0]),
            "classifier_label_encoding": label_encoding,
        }
        return reconstruction_encoder.ReconstructionLatentClassificationResult(
            train_latent=latent.train_latent,
            test_latent=latent.test_latent,
            predictions=predictions,
            probabilities=probabilities,
            classes=output_classes,
            encoder=latent.encoder,
            classifier=model,
            metadata=metadata,
        )

    setattr(fit_reconstruction_latent_classifier, _PATCH_MARKER, True)
    reconstruction_encoder.fit_reconstruction_latent_classifier = fit_reconstruction_latent_classifier


__all__ = ["install"]
