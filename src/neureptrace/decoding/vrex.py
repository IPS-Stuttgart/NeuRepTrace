"""Source-only variance-risk extrapolation for cross-subject decoding."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

VREX_PROTOCOL = "source_only_vrex_domain_generalization"
VREX_CATEGORY = "1_strict_source_only"
_MIN_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class VRExFitResult:
    """Fitted VREx classifier, target-free predictions, and provenance metadata."""

    model: "LinearVRExClassifier"
    probabilities: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class LinearVRExClassifier(ClassifierMixin, BaseEstimator):
    """Linear multiclass VREx classifier for strict source-only generalization."""

    def __init__(
        self,
        penalty_weight: float = 1.0,
        l2: float = 1e-4,
        max_iter: int = 500,
        tol: float = 1e-8,
        fit_intercept: bool = True,
        standardize: bool = True,
        class_weight: str | Mapping[Any, float] | None = "balanced",
    ):
        self.penalty_weight = penalty_weight
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.class_weight = class_weight

    def fit(
        self,
        source_features: Sequence[Sequence[float]] | np.ndarray,
        source_labels: Sequence[Any] | np.ndarray,
        *,
        source_domains: Sequence[Hashable] | np.ndarray,
    ) -> "LinearVRExClassifier":
        x = _feature_matrix(source_features, name="source_features")
        labels = _object_vector(source_labels, expected_length=x.shape[0], name="source_labels")
        domains = _object_vector(source_domains, expected_length=x.shape[0], name="source_domains")
        _validate_hashable(domains, name="source_domains")

        classes = tuple(dict.fromkeys(labels.tolist()))
        domain_values = tuple(dict.fromkeys(domains.tolist()))
        if len(classes) < 2:
            raise ValueError("VREx requires at least two source classes.")
        if len(domain_values) < 2:
            raise ValueError("VREx requires at least two source domains.")

        class_to_index = _unique_index(classes, name="source classes")
        domain_to_index = _unique_index(domain_values, name="source domains")
        y = np.asarray([class_to_index[value] for value in labels.tolist()], dtype=int)
        domain_ids = np.asarray([domain_to_index[value] for value in domains.tolist()], dtype=int)

        penalty_weight = _nonnegative_float(self.penalty_weight, name="penalty_weight")
        l2 = _nonnegative_float(self.l2, name="l2")
        max_iter = _positive_int(self.max_iter, name="max_iter")
        tol = _positive_float(self.tol, name="tol")
        sample_weights = _class_weights(labels, classes=classes, class_weight=self.class_weight)

        self.feature_mean_ = np.mean(x, axis=0) if self.standardize else np.zeros(x.shape[1], dtype=float)
        centered = x - self.feature_mean_
        ddof = 1 if x.shape[0] > 1 else 0
        self.feature_scale_ = np.maximum(np.std(centered, axis=0, ddof=ddof), _MIN_SCALE) if self.standardize else np.ones(x.shape[1], dtype=float)
        z = centered / self.feature_scale_

        self.classes_ = _object_vector(classes, expected_length=len(classes), name="classes")
        self.source_domains_ = _object_vector(domain_values, expected_length=len(domain_values), name="source_domains")
        self.n_features_in_ = int(x.shape[1])
        self.n_classes_ = len(classes)
        self.n_source_domains_ = len(domain_values)
        self.penalty_weight_ = penalty_weight
        self.l2_ = l2
        self.class_weight_vector_ = sample_weights.copy()

        n_free = self.n_classes_ - 1
        coefficient_size = self.n_features_in_ * n_free
        initial = np.zeros(coefficient_size + (n_free if self.fit_intercept else 0), dtype=float)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients, intercept = self._unpack(parameters)
            losses: list[float] = []
            gradients: list[np.ndarray] = []
            for domain_index in range(self.n_source_domains_):
                mask = domain_ids == domain_index
                domain_x = z[mask]
                domain_y = y[mask]
                weights = sample_weights[mask]
                weights = weights / np.sum(weights)
                probabilities = _softmax(_reference_logits(domain_x, coefficients, intercept))
                loss = float(np.sum(weights * -np.log(np.maximum(probabilities[np.arange(domain_y.shape[0]), domain_y], _MIN_SCALE))))
                residual = probabilities
                residual[np.arange(domain_y.shape[0]), domain_y] -= 1.0
                residual *= weights[:, None]
                pieces = [(domain_x.T @ residual[:, :-1]).ravel()]
                if self.fit_intercept:
                    pieces.append(np.sum(residual[:, :-1], axis=0))
                losses.append(loss)
                gradients.append(np.concatenate(pieces))

            loss_vector = np.asarray(losses, dtype=float)
            gradient_matrix = np.vstack(gradients)
            mean_loss = float(np.mean(loss_vector))
            centered_losses = loss_vector - mean_loss
            value = mean_loss + penalty_weight * float(np.mean(centered_losses**2))
            gradient = np.mean(gradient_matrix, axis=0) + 2.0 * penalty_weight * np.mean(centered_losses[:, None] * gradient_matrix, axis=0)
            if l2 > 0.0:
                value += 0.5 * l2 * float(np.sum(parameters[:coefficient_size] ** 2))
                gradient[:coefficient_size] += l2 * parameters[:coefficient_size]
            return value, gradient

        optimization = minimize(
            fun=lambda parameters: objective(parameters)[0],
            x0=initial,
            jac=lambda parameters: objective(parameters)[1],
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
        )
        if not np.all(np.isfinite(optimization.x)) or not np.isfinite(optimization.fun):
            raise RuntimeError("VREx optimization produced non-finite parameters or objective values.")

        self.coef_, self.intercept_ = self._unpack(np.asarray(optimization.x, dtype=float))
        self.optimization_success_ = bool(optimization.success)
        self.optimization_status_ = int(optimization.status)
        self.optimization_message_ = str(optimization.message)
        self.n_iter_ = int(getattr(optimization, "nit", 0))
        self.objective_ = float(optimization.fun)
        self.domain_risks_ = self._domain_risks(z, y, domain_ids, sample_weights)
        self.domain_risk_variance_ = float(np.var(self.domain_risks_))
        self.mean_domain_risk_ = float(np.mean(self.domain_risks_))
        return self

    def _unpack(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_free = self.n_classes_ - 1
        coefficient_size = self.n_features_in_ * n_free
        coefficients = parameters[:coefficient_size].reshape(self.n_features_in_, n_free)
        intercept = parameters[coefficient_size:] if self.fit_intercept else np.zeros(n_free, dtype=float)
        return coefficients, intercept

    def _standardized(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearVRExClassifier must be fitted before prediction.")
        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(f"features width {matrix.shape[1]} does not match fitted width {self.n_features_in_}.")
        return (matrix - self.feature_mean_) / self.feature_scale_

    def decision_function(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        logits = _reference_logits(self._standardized(features), self.coef_, self.intercept_)
        return logits[:, 1] - logits[:, 0] if self.n_classes_ == 2 else logits

    def predict_proba(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return _softmax(_reference_logits(self._standardized(features), self.coef_, self.intercept_))

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def _domain_risks(self, x: np.ndarray, y: np.ndarray, domains: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
        probabilities = _softmax(_reference_logits(x, self.coef_, self.intercept_))
        risks = []
        for domain_index in range(self.n_source_domains_):
            mask = domains == domain_index
            weights = sample_weights[mask]
            weights = weights / np.sum(weights)
            risks.append(float(np.sum(weights * -np.log(np.maximum(probabilities[mask, y[mask]], _MIN_SCALE)))))
        return np.asarray(risks, dtype=float)

    def metadata(self, *, test_rows: int | None = None) -> dict[str, Any]:
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearVRExClassifier must be fitted before metadata is available.")
        return {
            "vrex": True,
            "vrex_protocol": VREX_PROTOCOL,
            "vrex_protocol_category": VREX_CATEGORY,
            "vrex_uses_source_features": True,
            "vrex_uses_source_labels": True,
            "vrex_uses_source_domains": True,
            "vrex_uses_target_features": False,
            "vrex_uses_target_labels": False,
            "vrex_valid_for_strict_source_only": True,
            "vrex_n_source_domains": int(self.n_source_domains_),
            "vrex_n_classes": int(self.n_classes_),
            "vrex_n_features": int(self.n_features_in_),
            "vrex_test_rows": "" if test_rows is None else int(test_rows),
            "vrex_penalty_weight": float(self.penalty_weight_),
            "vrex_l2": float(self.l2_),
            "vrex_standardize": bool(self.standardize),
            "vrex_fit_intercept": bool(self.fit_intercept),
            "vrex_mean_domain_risk": float(self.mean_domain_risk_),
            "vrex_domain_risk_variance": float(self.domain_risk_variance_),
            "vrex_domain_risks": "|".join(f"{value:.12g}" for value in self.domain_risks_),
            "vrex_objective": float(self.objective_),
            "vrex_optimization_success": bool(self.optimization_success_),
            "vrex_optimization_status": int(self.optimization_status_),
            "vrex_optimization_message": self.optimization_message_,
            "vrex_n_iter": int(self.n_iter_),
        }


def fit_vrex_predict_proba(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    **model_kwargs: Any,
) -> VRExFitResult:
    model = LinearVRExClassifier(**model_kwargs)
    model.fit(source_features, source_labels, source_domains=source_domains)
    probabilities = model.predict_proba(test_features)
    return VRExFitResult(model=model, probabilities=probabilities, metadata=model.metadata(test_rows=probabilities.shape[0]))


def _reference_logits(x: np.ndarray, coefficients: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.column_stack([x @ coefficients + intercept, np.zeros(x.shape[0], dtype=float)])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, -50.0, 50.0))
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


def _labels_equal(left: object, right: object) -> bool:
    try:
        comparison = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(comparison, np.ndarray):
        try:
            return bool(np.all(comparison))
        except (TypeError, ValueError):
            return False
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False


def _class_weights(labels: np.ndarray, *, classes: Sequence[Any], class_weight: str | Mapping[Any, float] | None) -> np.ndarray:
    if class_weight is None:
        return np.ones(labels.shape[0], dtype=float)
    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be None, 'balanced', or a mapping keyed by source class.")
        weights = np.zeros(labels.shape[0], dtype=float)
        for class_label in classes:
            mask = np.asarray([_labels_equal(label, class_label) for label in labels.tolist()], dtype=bool)
            count = int(np.count_nonzero(mask))
            if count == 0:
                raise ValueError("source class has no matching rows for balanced class weighting.")
            weights[mask] = labels.shape[0] / (len(classes) * count)
        return weights
    if not isinstance(class_weight, Mapping):
        raise ValueError("class_weight must be None, 'balanced', or a mapping keyed by source class.")
    weights = np.asarray([float(class_weight.get(label, 1.0)) for label in labels.tolist()], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("class_weight values must be finite and positive.")
    return weights


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional feature matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _object_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    items = list(values)
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {len(items)} != {expected_length}.")
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _validate_hashable(values: np.ndarray, *, name: str) -> None:
    for value in values.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} values must be hashable; got {value!r}.") from exc


def _unique_index(values: Sequence[Hashable], *, name: str) -> dict[Hashable, int]:
    result: dict[Hashable, int] = {}
    for index, value in enumerate(values):
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be hashable; got {value!r}.") from exc
        if value in result:
            raise ValueError(f"{name} must be unique; duplicate value {value!r}.")
        result[value] = index
    return result


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed
