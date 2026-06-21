"""Reconstruction-loss latent representations for cross-subject decoding.

This module implements the protocol where an encoder/decoder is fit from an
unlabeled reconstruction objective and a supervised classifier is then trained
only on source labels in the latent space. Fitting the encoder on source rows
only is Protocol 1. Fitting it on source rows plus unlabeled target rows is
Protocol 2. Target labels are rejected by design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

RECONSTRUCTION_ENCODER_METHOD = "linear_reconstruction_encoder"
DEEP_MASKED_RECONSTRUCTION_ENCODER_METHOD = "deep_masked_reconstruction_encoder"
RECONSTRUCTION_LINEAR_ENCODER = "linear"
RECONSTRUCTION_MASKED_AUTOENCODER = "masked_autoencoder"
RECONSTRUCTION_ENCODER_KINDS = (RECONSTRUCTION_LINEAR_ENCODER, RECONSTRUCTION_MASKED_AUTOENCODER)
RECONSTRUCTION_SOURCE_ONLY = "source_only"
RECONSTRUCTION_SOURCE_PLUS_TARGET = "source_plus_target"
RECONSTRUCTION_FIT_SCOPES = (RECONSTRUCTION_SOURCE_ONLY, RECONSTRUCTION_SOURCE_PLUS_TARGET)
RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL = "strict_source_only"
RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL = "unlabeled_target_reconstruction"
DEFAULT_RECONSTRUCTION_COMPONENTS = 64
DEFAULT_MASKED_AUTOENCODER_HIDDEN_UNITS = (128, 64)
MIN_RECONSTRUCTION_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class ReconstructionEncoderConfig:
    """Configuration for reconstruction-loss latent encoders."""

    n_components: int | str | None = DEFAULT_RECONSTRUCTION_COMPONENTS
    fit_scope: str = RECONSTRUCTION_SOURCE_PLUS_TARGET
    standardize: bool = False
    encoder_kind: str = RECONSTRUCTION_LINEAR_ENCODER
    hidden_units: tuple[int, ...] = DEFAULT_MASKED_AUTOENCODER_HIDDEN_UNITS
    mask_fraction: float = 0.25
    noise_std: float = 0.0
    max_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    patience: int = 10
    dropout: float = 0.1
    classifier_max_iter: int = 1000
    classifier_C: float = 1.0
    classifier_class_weight: str | Mapping[Any, float] | None = None
    random_state: int | None = 13
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class ReconstructionLatentResult:
    """Latent train/test features and protocol metadata."""

    train_latent: np.ndarray
    test_latent: np.ndarray
    encoder: "LinearReconstructionEncoder | TorchMaskedReconstructionEncoder"
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReconstructionLatentClassificationResult:
    """Classifier outputs from reconstruction-latent features."""

    train_latent: np.ndarray
    test_latent: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray
    encoder: "LinearReconstructionEncoder | TorchMaskedReconstructionEncoder"
    classifier: BaseEstimator
    metadata: dict[str, Any] = field(default_factory=dict)


class LinearReconstructionEncoder:
    """Closed-form linear autoencoder/PCA fitted by reconstruction loss."""

    encoder_kind = RECONSTRUCTION_LINEAR_ENCODER
    representation_method = RECONSTRUCTION_ENCODER_METHOD

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

    def metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        return {
            "representation_encoder_kind": RECONSTRUCTION_LINEAR_ENCODER,
            "representation_is_nonlinear": False,
        }

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("LinearReconstructionEncoder must be fitted before use.")


class TorchMaskedReconstructionEncoder:
    """Nonlinear masked autoencoder fitted from unlabeled reconstruction loss.

    The model receives randomly masked/noised feature vectors and is optimized to
    reconstruct the original clean vector. ``transform`` then returns the latent
    code for clean feature vectors. The implementation intentionally has no label
    arguments, so target labels cannot affect the encoder fit.
    """

    encoder_kind = RECONSTRUCTION_MASKED_AUTOENCODER
    representation_method = DEEP_MASKED_RECONSTRUCTION_ENCODER_METHOD

    def __init__(
        self,
        n_components: int | str | None = DEFAULT_RECONSTRUCTION_COMPONENTS,
        *,
        standardize: bool = False,
        hidden_units: Sequence[int] | int = DEFAULT_MASKED_AUTOENCODER_HIDDEN_UNITS,
        mask_fraction: float = 0.25,
        noise_std: float = 0.0,
        max_epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.1,
        patience: int = 10,
        dropout: float = 0.1,
        random_state: int | None = 13,
        device: str = "auto",
    ):
        self.n_components = n_components
        self.standardize = standardize
        self.hidden_units = hidden_units
        self.mask_fraction = mask_fraction
        self.noise_std = noise_std
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state
        self.device = device

    def fit(self, features: Sequence[Sequence[float]] | np.ndarray):
        torch = _torch()
        x = _feature_matrix(features, name="reconstruction_features")
        fit_matrix = self._prepare_fit_matrix(x)
        random_state = None if self.random_state is None else _normalize_integer(self.random_state, name="random_state")
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

        hidden_units = _normalize_hidden_units(self.hidden_units)
        max_epochs = _normalize_integer(self.max_epochs, name="max_epochs", minimum=1)
        batch_size = _normalize_integer(self.batch_size, name="batch_size", minimum=1)
        patience = _normalize_integer(self.patience, name="patience", minimum=1)
        learning_rate = _normalize_positive_float(self.learning_rate, name="learning_rate")
        weight_decay = _normalize_nonnegative_float(self.weight_decay, name="weight_decay")
        mask_fraction = _normalize_bounded_float(self.mask_fraction, name="mask_fraction", lower=0.0, upper=1.0, inclusive_upper=False)
        noise_std = _normalize_nonnegative_float(self.noise_std, name="noise_std")
        validation_fraction = _normalize_bounded_float(self.validation_fraction, name="validation_fraction", lower=0.0, upper=1.0, inclusive_upper=False)
        dropout = _normalize_bounded_float(self.dropout, name="dropout", lower=0.0, upper=1.0, inclusive_upper=False)
        n_components = _effective_n_components(self.n_components, max_components=int(fit_matrix.shape[1]))

        device = self._resolve_device()
        model = _MaskedAutoencoderModule(input_dim=int(fit_matrix.shape[1]), latent_dim=int(n_components), hidden_units=hidden_units, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()

        fit_tensor = torch.as_tensor(fit_matrix.astype(np.float32, copy=False), dtype=torch.float32, device=device)
        rng = np.random.default_rng(random_state)
        all_indices = np.arange(fit_matrix.shape[0])
        train_idx, validation_idx = _train_validation_indices(all_indices, validation_fraction=validation_fraction, rng=rng)
        best_loss = np.inf
        best_state = None
        patience_left = patience
        epochs_run = 0
        final_training_loss = np.nan

        for epoch in range(max_epochs):
            epochs_run = epoch + 1
            model.train()
            epoch_losses: list[float] = []
            for batch_idx in _minibatch_indices(train_idx, batch_size=batch_size, rng=rng):
                clean_batch = fit_tensor[batch_idx]
                corrupted_batch = _corrupt_batch(clean_batch, mask_fraction=mask_fraction, noise_std=noise_std)
                optimizer.zero_grad(set_to_none=True)
                reconstructed, _latent = model(corrupted_batch)
                loss = loss_fn(reconstructed, clean_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            final_training_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan

            model.eval()
            with torch.no_grad():
                validation_clean = fit_tensor[validation_idx]
                validation_reconstructed, _latent = model(validation_clean)
                validation_loss = float(loss_fn(validation_reconstructed, validation_clean).detach().cpu())
            if validation_loss + 1e-7 < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.eval()
        self.device_ = device
        self.n_components_ = int(n_components)
        self.n_features_in_ = int(x.shape[1])
        self.n_fit_rows_ = int(x.shape[0])
        self.hidden_units_ = hidden_units
        self.mask_fraction_ = float(mask_fraction)
        self.noise_std_ = float(noise_std)
        self.max_epochs_ = int(max_epochs)
        self.batch_size_ = int(batch_size)
        self.learning_rate_ = float(learning_rate)
        self.weight_decay_ = float(weight_decay)
        self.validation_fraction_ = float(validation_fraction)
        self.patience_ = int(patience)
        self.dropout_ = float(dropout)
        self.n_epochs_ = int(epochs_run)
        self.best_validation_reconstruction_mse_ = float(best_loss)
        self.final_masked_training_reconstruction_mse_ = float(final_training_loss)
        self.reconstruction_mse_ = self.reconstruction_error(x)
        return self

    def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x = _feature_matrix(features, name="features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"features width {x.shape[1]} does not match fitted width {self.n_features_in_}.")
        matrix = ((x - self.mean_) / self.scale_).astype(np.float32, copy=False)
        return self._encode_matrix(matrix)

    def inverse_transform(self, latent: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        z = _feature_matrix(latent, name="latent")
        if z.shape[1] != self.n_components_:
            raise ValueError(f"latent width {z.shape[1]} does not match fitted latent width {self.n_components_}.")
        decoded = self._decode_matrix(z.astype(np.float32, copy=False))
        return decoded * self.scale_ + self.mean_

    def reconstruction_error(self, features: Sequence[Sequence[float]] | np.ndarray) -> float:
        x = _feature_matrix(features, name="features")
        return float(np.mean((x - self.inverse_transform(self.transform(x))) ** 2))

    def metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        return {
            "representation_encoder_kind": RECONSTRUCTION_MASKED_AUTOENCODER,
            "representation_is_nonlinear": True,
            "representation_masked_modeling": True,
            "representation_mask_fraction": float(self.mask_fraction_),
            "representation_noise_std": float(self.noise_std_),
            "representation_hidden_units": tuple(int(unit) for unit in self.hidden_units_),
            "representation_max_epochs": int(self.max_epochs_),
            "representation_epochs_run": int(self.n_epochs_),
            "representation_batch_size": int(self.batch_size_),
            "representation_learning_rate": float(self.learning_rate_),
            "representation_weight_decay": float(self.weight_decay_),
            "representation_validation_fraction": float(self.validation_fraction_),
            "representation_patience": int(self.patience_),
            "representation_dropout": float(self.dropout_),
            "representation_device": str(self.device_),
            "representation_validation_reconstruction_mse": float(self.best_validation_reconstruction_mse_),
            "representation_masked_training_reconstruction_mse": float(self.final_masked_training_reconstruction_mse_),
        }

    def _prepare_fit_matrix(self, x: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(x, axis=0)
        centered = x - self.mean_
        if self.standardize:
            variance = np.var(centered, axis=0, ddof=1 if x.shape[0] > 1 else 0)
            self.scale_ = np.sqrt(np.maximum(variance, MIN_RECONSTRUCTION_SCALE))
            fit_matrix = centered / self.scale_
        else:
            self.scale_ = np.ones(x.shape[1], dtype=float)
            fit_matrix = centered
        return fit_matrix

    def _resolve_device(self):
        torch = _torch()
        requested = str(self.device).strip().lower()
        if requested in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _encode_matrix(self, matrix: np.ndarray) -> np.ndarray:
        torch = _torch()
        outputs: list[np.ndarray] = []
        batch_size = int(getattr(self, "batch_size_", 128))
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, matrix.shape[0], batch_size):
                batch = torch.as_tensor(matrix[start : start + batch_size], dtype=torch.float32, device=self.device_)
                latent = self.model_.encode(batch)
                outputs.append(latent.detach().cpu().numpy().astype(float, copy=False))
        return np.vstack(outputs)

    def _decode_matrix(self, latent: np.ndarray) -> np.ndarray:
        torch = _torch()
        outputs: list[np.ndarray] = []
        batch_size = int(getattr(self, "batch_size_", 128))
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, latent.shape[0], batch_size):
                batch = torch.as_tensor(latent[start : start + batch_size], dtype=torch.float32, device=self.device_)
                decoded = self.model_.decode(batch)
                outputs.append(decoded.detach().cpu().numpy().astype(float, copy=False))
        return np.vstack(outputs)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMaskedReconstructionEncoder must be fitted before use.")


class _MaskedAutoencoderModule:
    def __new__(cls, *, input_dim: int, latent_dim: int, hidden_units: Sequence[int], dropout: float):
        torch = _torch()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                encoder_layers: list[Any] = []
                previous_dim = int(input_dim)
                for hidden_dim in hidden_units:
                    encoder_layers.append(torch.nn.Linear(previous_dim, int(hidden_dim)))
                    encoder_layers.append(torch.nn.ReLU())
                    if dropout > 0.0:
                        encoder_layers.append(torch.nn.Dropout(float(dropout)))
                    previous_dim = int(hidden_dim)
                encoder_layers.append(torch.nn.Linear(previous_dim, int(latent_dim)))
                self.encoder = torch.nn.Sequential(*encoder_layers)

                decoder_layers: list[Any] = []
                previous_dim = int(latent_dim)
                for hidden_dim in reversed(tuple(hidden_units)):
                    decoder_layers.append(torch.nn.Linear(previous_dim, int(hidden_dim)))
                    decoder_layers.append(torch.nn.ReLU())
                    if dropout > 0.0:
                        decoder_layers.append(torch.nn.Dropout(float(dropout)))
                    previous_dim = int(hidden_dim)
                decoder_layers.append(torch.nn.Linear(previous_dim, int(input_dim)))
                self.decoder = torch.nn.Sequential(*decoder_layers)

            def encode(self, features):
                return self.encoder(features)

            def decode(self, latent):
                return self.decoder(latent)

            def forward(self, features):
                latent = self.encode(features)
                return self.decode(latent), latent

        return Module()


def reconstruction_encoder_config(
    *,
    n_components: int | str | None = DEFAULT_RECONSTRUCTION_COMPONENTS,
    fit_scope: str | None = RECONSTRUCTION_SOURCE_PLUS_TARGET,
    standardize: bool = False,
    encoder_kind: str | None = RECONSTRUCTION_LINEAR_ENCODER,
    hidden_units: Sequence[int] | int = DEFAULT_MASKED_AUTOENCODER_HIDDEN_UNITS,
    mask_fraction: float = 0.25,
    noise_std: float = 0.0,
    max_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.1,
    patience: int = 10,
    dropout: float = 0.1,
    classifier_max_iter: int = 1000,
    classifier_C: float = 1.0,
    classifier_class_weight: str | Mapping[Any, float] | None = None,
    random_state: int | None = 13,
    device: str = "auto",
) -> ReconstructionEncoderConfig:
    """Normalize user-facing reconstruction-encoder options."""

    return ReconstructionEncoderConfig(
        n_components=_normalize_n_components_request(n_components),
        fit_scope=normalize_reconstruction_fit_scope(fit_scope),
        standardize=bool(standardize),
        encoder_kind=normalize_reconstruction_encoder_kind(encoder_kind),
        hidden_units=_normalize_hidden_units(hidden_units),
        mask_fraction=_normalize_bounded_float(mask_fraction, name="mask_fraction", lower=0.0, upper=1.0, inclusive_upper=False),
        noise_std=_normalize_nonnegative_float(noise_std, name="noise_std"),
        max_epochs=_normalize_integer(max_epochs, name="max_epochs", minimum=1),
        batch_size=_normalize_integer(batch_size, name="batch_size", minimum=1),
        learning_rate=_normalize_positive_float(learning_rate, name="learning_rate"),
        weight_decay=_normalize_nonnegative_float(weight_decay, name="weight_decay"),
        validation_fraction=_normalize_bounded_float(validation_fraction, name="validation_fraction", lower=0.0, upper=1.0, inclusive_upper=False),
        patience=_normalize_integer(patience, name="patience", minimum=1),
        dropout=_normalize_bounded_float(dropout, name="dropout", lower=0.0, upper=1.0, inclusive_upper=False),
        classifier_max_iter=_normalize_integer(classifier_max_iter, name="classifier_max_iter", minimum=1),
        classifier_C=_normalize_positive_float(classifier_C, name="classifier_C"),
        classifier_class_weight=classifier_class_weight,
        random_state=None if random_state is None else _normalize_integer(random_state, name="random_state"),
        device=str(device),
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


def normalize_reconstruction_encoder_kind(encoder_kind: str | None) -> str:
    """Normalize aliases for linear and deep masked reconstruction encoders."""

    normalized = RECONSTRUCTION_LINEAR_ENCODER if encoder_kind is None else str(encoder_kind).strip().lower().replace("-", "_")
    normalized = {
        "pca": RECONSTRUCTION_LINEAR_ENCODER,
        "linear_autoencoder": RECONSTRUCTION_LINEAR_ENCODER,
        "linear_reconstruction_encoder": RECONSTRUCTION_LINEAR_ENCODER,
        "deep": RECONSTRUCTION_MASKED_AUTOENCODER,
        "nonlinear": RECONSTRUCTION_MASKED_AUTOENCODER,
        "masked": RECONSTRUCTION_MASKED_AUTOENCODER,
        "masked_modeling": RECONSTRUCTION_MASKED_AUTOENCODER,
        "deep_masked": RECONSTRUCTION_MASKED_AUTOENCODER,
        "deep_masked_autoencoder": RECONSTRUCTION_MASKED_AUTOENCODER,
        "torch_masked_autoencoder": RECONSTRUCTION_MASKED_AUTOENCODER,
        "mae": RECONSTRUCTION_MASKED_AUTOENCODER,
    }.get(normalized, normalized)
    if normalized not in RECONSTRUCTION_ENCODER_KINDS:
        raise ValueError(f"Unknown reconstruction encoder kind {encoder_kind!r}. Available encoders: {', '.join(RECONSTRUCTION_ENCODER_KINDS)}.")
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

    encoder = _build_reconstruction_encoder(cfg).fit(fit_matrix)
    train_latent = encoder.transform(train_matrix)
    test_latent = encoder.transform(test_matrix)
    protocol = RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL if uses_unlabeled_target else RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL
    metadata = {
        "representation_method": encoder.representation_method,
        "representation_fit_scope": cfg.fit_scope,
        "representation_protocol": protocol,
        "representation_protocol_note": (
            "uses source rows plus unlabeled target features for reconstruction; category-2 target-adaptive representation"
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
        **encoder.metadata(),
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


def _build_reconstruction_encoder(cfg: ReconstructionEncoderConfig) -> LinearReconstructionEncoder | TorchMaskedReconstructionEncoder:
    if cfg.encoder_kind == RECONSTRUCTION_LINEAR_ENCODER:
        return LinearReconstructionEncoder(n_components=cfg.n_components, standardize=cfg.standardize)
    if cfg.encoder_kind == RECONSTRUCTION_MASKED_AUTOENCODER:
        return TorchMaskedReconstructionEncoder(
            n_components=cfg.n_components,
            standardize=cfg.standardize,
            hidden_units=cfg.hidden_units,
            mask_fraction=cfg.mask_fraction,
            noise_std=cfg.noise_std,
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            validation_fraction=cfg.validation_fraction,
            patience=cfg.patience,
            dropout=cfg.dropout,
            random_state=cfg.random_state,
            device=cfg.device,
        )
    raise ValueError(f"Unknown reconstruction encoder kind {cfg.encoder_kind!r}.")


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


def _normalize_hidden_units(value: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
        if not text:
            return tuple()
        raw_values: Sequence[Any] = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        raw_values = [value]
    else:
        raw_values = list(value)
    return tuple(_normalize_integer(unit, name="hidden_units", minimum=1) for unit in raw_values)


def _train_validation_indices(indices: np.ndarray, *, validation_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    if shuffled.shape[0] < 4 or validation_fraction <= 0.0:
        return shuffled, shuffled
    validation_size = int(round(float(validation_fraction) * shuffled.shape[0]))
    validation_size = min(max(validation_size, 1), shuffled.shape[0] - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


def _minibatch_indices(indices: np.ndarray, *, batch_size: int, rng: np.random.Generator):
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    for start in range(0, shuffled.shape[0], batch_size):
        yield shuffled[start : start + batch_size]


def _corrupt_batch(clean_batch, *, mask_fraction: float, noise_std: float):
    torch = _torch()
    corrupted = clean_batch.clone()
    if mask_fraction > 0.0:
        keep_mask = torch.rand_like(corrupted) >= float(mask_fraction)
        corrupted = corrupted * keep_mask.to(dtype=corrupted.dtype)
    if noise_std > 0.0:
        corrupted = corrupted + float(noise_std) * torch.randn_like(corrupted)
    return corrupted


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("The masked reconstruction encoder requires torch, e.g. `pip install neureptrace[torch]`.") from exc
    return torch


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


def _normalize_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative finite value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite value.") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return numeric


def _normalize_bounded_float(value: Any, *, name: str, lower: float, upper: float, inclusive_upper: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite in the requested range.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite in the requested range.") from exc
    upper_ok = numeric <= upper if inclusive_upper else numeric < upper
    if not np.isfinite(numeric) or numeric < lower or not upper_ok:
        comparator = "<=" if inclusive_upper else "<"
        raise ValueError(f"{name} must satisfy {lower} <= {name} {comparator} {upper}.")
    return numeric
