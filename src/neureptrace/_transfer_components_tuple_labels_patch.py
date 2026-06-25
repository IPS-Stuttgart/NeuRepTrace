"""Runtime patch for atomic labels in transfer-component classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from neureptrace.decoding import transfer_components as _tc

_ORIGINAL = None


def install() -> None:
    """Install the tuple-label safe TCA classifier wrapper."""

    global _ORIGINAL  # noqa: PLW0603
    if _ORIGINAL is not None:
        return
    _ORIGINAL = _tc.fit_transfer_component_classifier
    _tc.fit_transfer_component_classifier = _fit_transfer_component_classifier_atomic_labels


# pylint: disable-next=too-many-arguments,too-many-locals
def _fit_transfer_component_classifier_atomic_labels(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    config: _tc.TransferComponentConfig | Mapping[str, Any] | None = None,
    classifier: BaseEstimator | None = None,
    classifier_C: float | str = 1.0,
    classifier_max_iter: int | str = 1000,
    classifier_class_weight: str | Mapping[Any, float] | None = "balanced",
    sample_weight: Sequence[float] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
) -> _tc.TransferComponentClassificationResult:
    """Fit TCA and decode atomic source labels back after sklearn fitting."""

    if target_labels is not None:
        raise ValueError("TCA classification does not accept target_labels; target labels must be reserved for scoring.")
    labels = _tc._label_vector(source_labels, expected_length=_tc._feature_matrix(source_features, name="source_features").shape[0], name="source_labels")
    encoded_labels, original_classes = _encode_labels(labels)
    if original_classes.shape[0] < 2:
        raise ValueError("source_labels must contain at least two classes.")
    tca = _tc.fit_transfer_component_features(source_features=source_features, target_features=target_features, config=config)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights is not None:
        if weights.shape[0] != labels.shape[0]:
            raise ValueError(f"sample_weight must contain one value per source row: {weights.shape[0]} != {labels.shape[0]}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weight must contain finite non-negative values.")
    encoded_class_weight = _encode_class_weight_mapping(classifier_class_weight, original_classes)
    model = clone(classifier) if classifier is not None else LogisticRegression(
        C=_tc._positive_float(classifier_C, name="classifier_C"),
        max_iter=_tc._positive_int(classifier_max_iter, name="classifier_max_iter"),
        class_weight=encoded_class_weight,
        random_state=13,
    )
    fit_kwargs = {} if weights is None else {"sample_weight": weights}
    model.fit(tca.source_features, encoded_labels, **fit_kwargs)
    model_class_indices = np.asarray(getattr(model, "classes_", np.arange(original_classes.shape[0])))
    predictions = _decode_label_indices(np.asarray(model.predict(tca.target_features)), original_classes)
    probabilities = _tc._predict_probabilities_or_none(model, tca.target_features)
    metadata = {
        **tca.metadata,
        "transfer_component_classifier": type(model).__name__,
        "transfer_component_classifier_uses_source_labels": True,
        "transfer_component_classifier_uses_target_labels": False,
    }
    return _tc.TransferComponentClassificationResult(
        source_features=tca.source_features,
        target_features=tca.target_features,
        predictions=predictions,
        probabilities=probabilities,
        classes=_decode_label_indices(model_class_indices, original_classes),
        classifier=model,
        tca=tca,
        metadata=metadata,
    )


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes: list[Any] = []
    encoded = np.empty(labels.shape[0], dtype=np.int64)
    for index, label in enumerate(labels.tolist()):
        for class_index, class_label in enumerate(classes):
            if _values_equal(label, class_label):
                encoded[index] = class_index
                break
        else:
            classes.append(label)
            encoded[index] = len(classes) - 1
    return encoded, _object_vector(classes)


def _encode_class_weight_mapping(class_weight: str | Mapping[Any, float] | None, classes: np.ndarray) -> str | dict[int, float] | None:
    if not isinstance(class_weight, Mapping):
        return class_weight
    encoded: dict[int, float] = {}
    for class_index, class_label in enumerate(classes.tolist()):
        for key, weight in class_weight.items():
            if _values_equal(key, class_label):
                encoded[class_index] = float(weight)
                break
    return encoded


def _decode_label_indices(values: Sequence[Any] | np.ndarray, classes: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    decoded = np.empty(array.shape, dtype=object)
    for index, value in np.ndenumerate(array):
        class_index = int(value)
        if class_index < 0 or class_index >= classes.shape[0]:
            raise ValueError(f"Classifier returned unknown encoded class index {value!r}.")
        decoded[index] = classes[class_index]
    return decoded.reshape(array.shape)


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
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
