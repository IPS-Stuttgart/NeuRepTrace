"""Reconstruction-loss latent representations for cross-subject decoding.

This module implements the protocol where an encoder/decoder is fit from an
unlabeled reconstruction objective and a supervised classifier is then trained
only on source labels in the latent space.  Fitting the encoder on source rows
only is Protocol 1.  Fitting it on source rows plus unlabeled target rows is
Protocol 2.  Target labels are rejected by design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

RECONSTRUCTION_ENCODER_METHOD = "linear_reconstruction_encoder"
RECONSTRUCTION_SOURCE_ONLY = "source_only"
RECONSTRUCTION_SOURCE_PLUS_TARGET = "source_plus_target"
RECONSTRUCTION_FIT_SCOPES = (RECONSTRUCTION_SOURCE_ONLY, RECONSTRUCTION_SOURCE_PLUS_TARGET)
RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL = "strict_source_only"
RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL = "unlabeled_target_reconstruction"
DEFAULT_RECONSTRUCTION_COMPONENTS = 64
MIN_RECONSTRUCTION_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class ReconstructionEncoderConfig:
    """Configuration for the linear reconstruction encoder."""

    n_components: int | str | None = DEFAULT_RECONSTRUCTION_COMPONENTS
    fit_scope: str = RECONSTRUCTION_SOURCE_PLUS_TARGET
    standardize: bool = False
    classifier_max_iter: int = 1000
    classifier_C: float = 1.0
    classifier_class_weight: str | Mapping[Any, float] | None = None
    random_state: int | None = 13


@dataclass(frozen=True, slots=True)
class ReconstructionLatentResult:
    """Latent train/test features and protocol metadata."""

    train_latent: np.ndarray
    test_latent: np.ndarray
    encoder: "LinearReconstructionEncoder"
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReconstructionLatentClassificationResult:
    """Classifier outputs from reconstruction-latent features."""

    train_latent: np.ndarray
    test_latent: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray
    encoder: "LinearReconstructionEncoder"
    classifier: BaseEstimator
    metadata: dict[str, Any] = field(default_factory=dict)


class LinearReconstructionEncoder:
    """Closed-form linear autoencoder/PCA fitted by reconstruction loss."""

    def __init__(self, n_components: int | str | None = DEFAULT_RECONSTRUCTION_COMPONENTS, *, standardize: bool = False):
        self.n_components = n_components
        self.standardize = standardize

    def fit(self, features: Sequence[Sequence[float]] | np.ndarray):
        x = _feature_matrix(features, name="reconstruction_features")
        self.mean_ = np.mean(x, axis=0)
        centered = x - self.mean_
        if self.standardize:
            variance = np.var(centered, axis=0, ddof=1 if x.shape[0] > 1 else 0)
            self.scale_ = np.sqrt(np.maximum(variance, MIN_RECONSTRUCTION_SCALE))
            fit_matrix = centered / self.scale_
        else:
            self.scale_ = np.ones(x.shape[1], dtype=float)
            fit_matrix = centered
        max_components = max(1, min(int(fit_matrix.shape[0]), int(fit_matrix.shape[1])))
        n_components = _effective_n_components(self.n_components, max_components=max_components)
        _u, singular_values, vt = np.linalg.svd(fit_matrix, full_matrices=False)
        self.components_ = vt[:n_components]
        self.singular_values_ = singular_values[:n_components]
        energy = float(np.sum(singular_values**2))
        self.explained_variance_ratio_ = np.zeros(n_components, dtype=float) if energy <= 0.0 else (singular_values[:n_components] ** 2) / energy
        self.n_components_ = int(n_components)
        self.n_features_in_ = int(x.shape[1])
        self.n_fit_rows_ = int(x.shape[0])
        self.reconstruction_mse_ = self.reconstruction_error(x)
        return self

    def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x = _feature_matrix(features, name="features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"features width {x.shape[1]} does not match fitted width {self.n_features_in_}.")
        return ((x - self.mean_) / self.scale_) @ self.components_.T

    def inverse_transform(self, latent: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        z = _feature_matrix(latent, name="latent")
        if z.shape[1] != self.n_components_:
            raise ValueError(f"latent width {z.shape[1]} does not match fitted latent width {self.n_components_}.")
        return (z @ self.components_) * self.scale_ + self.mean_

    def reconstruction_error(self, features: Sequence[Sequence[float]] | np.ndarray) -> float:
        x = _feature_matrix(features, name="features")
        return float(np.mean((x - self.inverse_transform(self.transform(x))) ** 2))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("LinearReconstructionEncoder must be fitted before use.")


def reconstruction_encoder_config(
    *,
    n_components: int | str | None = DEFAULT_RECONSTRUCTION_COMPONENTS,
    fit_scope: str | None = RECONSTRUCTION_SOURCE_PLUS_TARGET,
    standardize: bool = False,
    classifier_max_iter: int = 1000,
    classifier_C: float = 1.0,
    classifier_class_weight: str | Mapping[Any, float] | None = None,
    random_state: int | None = 13,
) -> ReconstructionEncoderConfig:
    """Normalize user-facing reconstruction-encoder options."""

    return ReconstructionEncoderConfig(
        n_components=_normalize_n_components_request(n_components),
        fit_scope=normalize_reconstruction_fit_scope(fit_scope),
        standardize=bool(standardize),
        classifier_max_iter=_normalize_integer(classifier_max_iter, name="classifier_max_iter", minimum=1),
        classifier_C=_normalize_positive_float(classifier_C, name="classifier_C"),
        classifier_class_weight=classifier_class_weight,
        random_state=None if random_state is None else _normalize_integer(random_state, name="random_state"),
    )


def normalize_reconstruction_fit_scope(fit_scope: str | None) -> str:
    """Normalize aliases for source-only and source-plus-target encoder fits."""

    normalized = RECONSTRUCTION_SOURCE_PLUS_TARGET if fit_scope is None else str(fit_scope).strip().lower().replace("-", "_")
    normalized = {
        "source": RECONSTRUCTION_SOURCE_ONLY,
        "sourceonly": RECONSTRUCTION_SOURCE_ONLY,
        "strict_source_only": RECONSTRUCTION_SOURCE_ONLY,
        "category_1": RECONSTRUCTION_SOURCE_ONLY,
        "protocol_1": RECONSTRUCTION_SOURCE_ONLY,
        "all_data": RECONSTRUCTION_SOURCE_PLUS_TARGET,
        "source_target": RECONSTRUCTION_SOURCE_PLUS_TARGET,
        "source_and_target": RECONSTRUCTION_SOURCE_PLUS_TARGET,
        "target_adaptive": RECONSTRUCTION_SOURCE_PLUS_TARGET,
        "unlabeled_target": RECONSTRUCTION_SOURCE_PLUS_TARGET,
        "category_2": RECONSTRUCTION_SOURCE_PLUS_TARGET,
        "protocol_2": RECONSTRUCTION_SOURCE_PLUS_TARGET,
    }.get(normalized, normalized)
    if normalized not in RECONSTRUCTION_FIT_SCOPES:
        raise ValueError(f"Unknown reconstruction fit scope {fit_scope!r}. Available scopes: {', '.join(RECONSTRUCTION_FIT_SCOPES)}.")
    return normalized


def fit_reconstruction_latent_space(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: ReconstructionEncoderConfig | None = None,
    target_encoder_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
) -> ReconstructionLatentResult:
    """Fit the reconstruction encoder and return latent train/test features."""

    if target_labels is not None:
        raise ValueError("Reconstruction latent protocols do not accept target labels.")
    cfg = reconstruction_encoder_config() if config is None else config
    train_matrix = _feature_matrix(train_features, name="train_features")
    test_matrix = _feature_matrix(test_features, name="test_features")
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError(f"train_features and test_features must have the same feature width: {train_matrix.shape[1]} != {test_matrix.shape[1]}.")

    if cfg.fit_scope == RECONSTRUCTION_SOURCE_ONLY:
        if target_encoder_features is not None:
            raise ValueError("source_only reconstruction does not accept target_encoder_features.")
        fit_matrix = train_matrix
        uses_unlabeled_target = False
        target_source = ""
    else:
        target_matrix = test_matrix if target_encoder_features is None else _feature_matrix(target_encoder_features, name="target_encoder_features")
        if target_matrix.shape[1] != train_matrix.shape[1]:
            raise ValueError(f"target_encoder_features and train_features must have the same feature width: {target_matrix.shape[1]} != {train_matrix.shape[1]}.")
        fit_matrix = np.vstack([train_matrix, target_matrix])
        uses_unlabeled_target = True
        target_source = "test_features_transductive" if target_encoder_features is None else "target_encoder_features"

    encoder = LinearReconstructionEncoder(n_components=cfg.n_components, standardize=cfg.standardize).fit(fit_matrix)
    train_latent = encoder.transform(train_matrix)
    test_latent = encoder.transform(test_matrix)
    protocol = RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL if uses_unlabeled_target else RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL
    metadata = {
        "representation_method": RECONSTRUCTION_ENCODER_METHOD,
        "representation_fit_scope": cfg.fit_scope,
        "representation_protocol": protocol,
        "representation_protocol_note": (
            "uses source rows plus unlabeled target features for reconstruction; "
            "category-2 target-adaptive representation"
            if uses_unlabeled_target
            else "fits reconstruction encoder on source rows only; strict source-only representation"
        ),
        "representation_uses_unlabeled_target_data": uses_unlabeled_target,
        "representation_target_labels_used": False,
        "representation_valid_for_strict_source_only": not uses_unlabeled_target,
        "representation_valid_for_benchmark": not uses_unlabeled_target,
        "representation_target_feature_source": target_source,
        "representation_requested_components": cfg.n_components,
        "representation_n_components": int(encoder.n_components_),
        "representation_feature_dim": int(train_matrix.shape[1]),
        "representation_train_rows": int(train_matrix.shape[0]),
        "representation_test_rows": int(test_matrix.shape[0]),
        "representation_fit_rows": int(fit_matrix.shape[0]),
        "representation_standardized": bool(cfg.standardize),
        "representation_train_reconstruction_mse": encoder.reconstruction_error(train_matrix),
        "representation_test_reconstruction_mse": encoder.reconstruction_error(test_matrix),
        "representation_fit_reconstruction_mse": float(encoder.reconstruction_mse_),
    }
    return ReconstructionLatentResult(train_latent=train_latent, test_latent=test_latent, encoder=encoder, metadata=metadata)


def fit_reconstruction_latent_classifier(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: ReconstructionEncoderConfig | None = None,
    target_encoder_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
    classifier: BaseEstimator | None = None,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> ReconstructionLatentClassificationResult:
    """Train a source-label classifier in the reconstruction latent space."""

    if target_labels is not None:
        raise ValueError("Reconstruction latent classifier does not accept target labels.")
    cfg = reconstruction_encoder_config() if config is None else config
    y = np.asarray(train_labels).reshape(-1)
    latent = fit_reconstruction_latent_space(
        train_features=train_features,
        test_features=test_features,
        config=cfg,
        target_encoder_features=target_encoder_features,
    )
    if latent.train_latent.shape[0] != y.shape[0]:
        raise ValueError("train_features and train_labels must contain the same number of rows.")
    if np.unique(y).shape[0] < 2:
        raise ValueError("train_labels must contain at least two classes.")

    model = clone(classifier) if classifier is not None else LogisticRegression(
        C=cfg.classifier_C,
        class_weight=cfg.classifier_class_weight,
        max_iter=cfg.classifier_max_iter,
        random_state=cfg.random_state,
    )
    fit_kwargs = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight, dtype=float)}
    model.fit(latent.train_latent, y, **fit_kwargs)
    probabilities = np.asarray(model.predict_proba(latent.test_latent), dtype=float) if hasattr(model, "predict_proba") else None
    classes = np.asarray(getattr(model, "classes_", np.unique(y)))
    metadata = {
        **latent.metadata,
        "classifier_label_source": "source_train_labels",
        "classifier_target_labels_used": False,
        "classifier_name": type(model).__name__,
        "classifier_n_classes": int(classes.shape[0]),
    }
    return ReconstructionLatentClassificationResult(
        train_latent=latent.train_latent,
        test_latent=latent.test_latent,
        predictions=np.asarray(model.predict(latent.test_latent)),
        probabilities=probabilities,
        classes=classes,
        encoder=latent.encoder,
        classifier=model,
        metadata=metadata,
    )


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _normalize_n_components_request(value: int | str | None) -> int | str:
    if value is None:
        return DEFAULT_RECONSTRUCTION_COMPONENTS
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return DEFAULT_RECONSTRUCTION_COMPONENTS
        if text in {"all", "full", "inf", "infinity"}:
            return "all"
        value = text
    return int(_normalize_integer(value, name="n_components", minimum=1))


def _effective_n_components(value: int | str | None, *, max_components: int) -> int:
    requested = _normalize_n_components_request(value)
    return int(max_components) if requested == "all" else min(int(requested), int(max_components))


def _normalize_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    integer = int(numeric)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer


def _normalize_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value.") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return numeric
