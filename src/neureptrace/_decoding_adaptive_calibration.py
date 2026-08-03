from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from neureptrace._object_label_utils import values_equal

_MARKER = "_neureptrace_adaptive_calibration_installed"


def _sample_weight_has_boolean_values(sample_weight) -> bool:
    values = np.asarray(sample_weight, dtype=object).reshape(-1)
    return any(isinstance(value, (bool, np.bool_)) for value in values)


def _normalize_sample_weight(sample_weight, *, n_rows: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    if _sample_weight_has_boolean_values(sample_weight):
        raise ValueError("sample_weight must contain finite non-negative numeric values, not booleans.")
    try:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight must contain finite non-negative numeric values.") from exc
    if weights.shape[0] != int(n_rows):
        raise ValueError("sample_weight must contain one weight per label.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weight must contain finite non-negative numeric values.")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("sample_weight must have positive total weight.")
    return weights


def _validate_cv(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Calibration cv must be an integer at least 2.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration cv must be an integer at least 2.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError("Calibration cv must be an integer at least 2.")
    requested = int(numeric)
    if requested < 2:
        raise ValueError("Calibration cv must be an integer at least 2.")
    return requested


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _atomic_label_vector(labels: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    """Return one scalar or composite label object per feature row."""

    array = np.asarray(labels, dtype=object)
    if array.ndim == 0:
        vector = _object_vector([array.item()])
    elif array.ndim == 1:
        if array.shape[0] == expected_length:
            vector = _object_vector(array.tolist())
        elif expected_length == 1:
            vector = _object_vector([tuple(array.tolist())])
        else:
            vector = _object_vector(array.reshape(-1).tolist())
    else:
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            vector = _object_vector(rows[:, 0].tolist())
        else:
            vector = _object_vector([tuple(row.tolist()) for row in rows])

    if vector.shape[0] != expected_length:
        raise ValueError(
            "labels must contain one label per feature row: "
            f"{vector.shape[0]} != {expected_length}."
        )
    return vector


def _is_composite_label(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (tuple, list, dict))


def _encode_composite_labels(
    labels: Sequence[Any] | np.ndarray,
    *,
    expected_length: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Dense-encode composite labels for sklearn while preserving public classes."""

    label_vector = _atomic_label_vector(labels, expected_length=expected_length)
    if not any(_is_composite_label(label) for label in label_vector.tolist()):
        return None

    classes: list[object] = []
    encoded = np.empty(label_vector.shape[0], dtype=np.int64)
    for row_index, label in enumerate(label_vector.tolist()):
        for class_index, class_label in enumerate(classes):
            if values_equal(label, class_label):
                encoded[row_index] = class_index
                break
        else:
            encoded[row_index] = len(classes)
            classes.append(label)
    return encoded, _object_vector(classes)


def _fit_with_optional_sample_weight(model, features, labels, sample_weight=None):
    """Fit an estimator and route sample weights through sklearn pipelines.

    The adaptive calibration fallback can replace ``CalibratedClassifierCV`` with
    the raw estimator when a fold has too few examples per class. Many NeuRepTrace
    decoders are sklearn pipelines, and those require ``<final_step>__sample_weight``
    rather than a top-level ``sample_weight`` argument. Routing here keeps tiny,
    sample-weighted folds usable instead of failing after the fallback decision.
    """

    weights = _normalize_sample_weight(sample_weight, n_rows=len(labels))
    if weights is None:
        model.fit(features, labels)
        return model

    try:
        model.fit(features, labels, sample_weight=weights)
    except (TypeError, ValueError):
        if not hasattr(model, "steps") or not getattr(model, "steps"):
            raise
        final_step_name = model.steps[-1][0]
        model.fit(features, labels, **{f"{final_step_name}__sample_weight": weights})
    return model


class AdaptiveCalibratedClassifierCV(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator: Any, method: str = "sigmoid", cv: int = 3):
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def fit(self, features, labels, sample_weight=None):
        try:
            n_rows = len(features)
        except TypeError:
            n_rows = len(labels)

        composite_encoding = _encode_composite_labels(labels, expected_length=n_rows)
        if composite_encoding is None:
            labels_for_fit = np.asarray(labels)
            public_classes = None
        else:
            labels_for_fit, public_classes = composite_encoding

        classes, counts = np.unique(labels_for_fit, return_counts=True)
        if classes.shape[0] < 2:
            raise ValueError("Calibration requires at least two classes.")
        requested = _validate_cv(self.cv)
        min_count = int(counts.min())
        self.classes_ = classes if public_classes is None else public_classes
        self.internal_classes_ = classes
        self.composite_label_encoding_ = public_classes is not None
        self.requested_calibration_cv_ = requested
        self.min_class_count_ = min_count
        if min_count >= 2:
            self.calibration_cv_ = min(requested, min_count)
            self.used_uncalibrated_fallback_ = False
            self.model_ = self._calibration_factory(clone(self.estimator), method=self.method, cv=self.calibration_cv_)
        else:
            self.calibration_cv_ = 0
            self.used_uncalibrated_fallback_ = True
            self.model_ = clone(self.estimator)
        _fit_with_optional_sample_weight(self.model_, features, labels_for_fit, sample_weight)
        if public_classes is None and hasattr(self.model_, "classes_"):
            self.classes_ = np.asarray(self.model_.classes_)
        if hasattr(self.model_, "classes_"):
            self.internal_classes_ = np.asarray(self.model_.classes_)
        if hasattr(self.model_, "n_features_in_"):
            self.n_features_in_ = self.model_.n_features_in_
        return self

    def _model(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("Estimator must be fitted before prediction.")
        return self.model_

    def _public_predictions(self, predictions) -> np.ndarray:
        prediction_array = np.asarray(predictions)
        if not getattr(self, "composite_label_encoding_", False):
            return prediction_array

        internal_classes = np.asarray(self.internal_classes_)
        positions = {class_label: index for index, class_label in enumerate(internal_classes.tolist())}
        try:
            indices = [positions[label] for label in prediction_array.reshape(-1).tolist()]
        except (KeyError, TypeError) as exc:  # pragma: no cover - defensive corrupt-estimator guard
            raise RuntimeError("Calibrated estimator returned an unknown encoded class label.") from exc
        return self.classes_[np.asarray(indices, dtype=int)]

    def predict_proba(self, features):
        model = self._model()
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            if probabilities.ndim == 2:
                return probabilities
        from neureptrace.decoding import score_to_probabilities

        if hasattr(model, "decision_function"):
            return score_to_probabilities(model.decision_function(features))
        predictions = np.asarray(model.predict(features))
        probabilities = np.zeros((predictions.shape[0], self.classes_.shape[0]), dtype=float)
        internal_classes = np.asarray(self.internal_classes_)
        lookup = {label: index for index, label in enumerate(internal_classes.tolist())}
        for row_index, label in enumerate(predictions.tolist()):
            probabilities[row_index, lookup[label]] = 1.0
        return probabilities

    def decision_function(self, features):
        model = self._model()
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(features), dtype=float)
        probabilities = np.clip(self.predict_proba(features), 1e-12, 1.0)
        if probabilities.shape[1] == 2:
            return np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])
        return np.log(probabilities)

    def predict(self, features):
        model = self._model()
        if hasattr(model, "predict"):
            return self._public_predictions(model.predict(features))
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


def install() -> None:
    from neureptrace import decoding

    if getattr(decoding, _MARKER, False):
        return
    original_factory = decoding._make_calibrated_classifier
    AdaptiveCalibratedClassifierCV._calibration_factory = staticmethod(original_factory)

    def _make_calibrated_classifier(estimator, *, method: str, cv: int):
        return AdaptiveCalibratedClassifierCV(estimator=estimator, method=method, cv=cv)

    decoding.AdaptiveCalibratedClassifierCV = AdaptiveCalibratedClassifierCV
    decoding._make_calibrated_classifier = _make_calibrated_classifier
    setattr(decoding, _MARKER, True)
