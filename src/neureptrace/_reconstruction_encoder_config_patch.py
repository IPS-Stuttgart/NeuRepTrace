"""Normalize reconstruction-encoder config values consistently."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_reconstruction_encoder_config_patch_installed"
_HIDDEN_UNITS_PATCH_MARKER = "_neureptrace_reconstruction_hidden_units_patch_installed"
_FIT_SCOPE_PATCH_MARKER = "_neureptrace_reconstruction_fit_scope_patch_installed"
_CLASSIFIER_CONFIG_PATCH_MARKER = "_neureptrace_reconstruction_classifier_config_patch_installed"


def _normalize_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        integer = int(value)
        if integer in {0, 1}:
            return bool(integer)
        raise ValueError(f"{name} must be a boolean value.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _normalize_config(reconstruction_encoder: Any, config: Any) -> Any:
    """Run direct dataclass configs through the public normalization factory."""

    return reconstruction_encoder.reconstruction_encoder_config(
        n_components=config.n_components,
        fit_scope=config.fit_scope,
        standardize=config.standardize,
        encoder_kind=config.encoder_kind,
        hidden_units=config.hidden_units,
        mask_fraction=config.mask_fraction,
        noise_std=config.noise_std,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        validation_fraction=config.validation_fraction,
        patience=config.patience,
        dropout=config.dropout,
        classifier_max_iter=config.classifier_max_iter,
        classifier_C=config.classifier_C,
        classifier_class_weight=config.classifier_class_weight,
        random_state=config.random_state,
        device=config.device,
    )


def _install_hidden_units_normalizer(reconstruction_encoder: Any) -> None:
    original = reconstruction_encoder._normalize_hidden_units
    if getattr(original, _HIDDEN_UNITS_PATCH_MARKER, False):
        return

    @wraps(original)
    def _normalize_hidden_units(value: Any) -> tuple[int, ...]:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("hidden_units must be an integer or sequence of integers.")
        if isinstance(value, (float, np.floating)):
            return (reconstruction_encoder._normalize_integer(value, name="hidden_units", minimum=1),)
        try:
            return original(value)
        except TypeError as exc:
            raise ValueError("hidden_units must be an integer or sequence of integers.") from exc

    setattr(_normalize_hidden_units, _HIDDEN_UNITS_PATCH_MARKER, True)
    reconstruction_encoder._normalize_hidden_units = _normalize_hidden_units


def _install_fit_scope_normalizer(reconstruction_encoder: Any) -> None:
    original = reconstruction_encoder.fit_reconstruction_latent_space
    if getattr(original, _FIT_SCOPE_PATCH_MARKER, False):
        return

    @wraps(original)
    def fit_reconstruction_latent_space(*, config=None, **kwargs: Any):
        if config is not None:
            config = _normalize_config(reconstruction_encoder, config)
        return original(config=config, **kwargs)

    setattr(fit_reconstruction_latent_space, _FIT_SCOPE_PATCH_MARKER, True)
    reconstruction_encoder.fit_reconstruction_latent_space = fit_reconstruction_latent_space


def _install_classifier_config_normalizer(reconstruction_encoder: Any) -> None:
    original = reconstruction_encoder.fit_reconstruction_latent_classifier
    if getattr(original, _CLASSIFIER_CONFIG_PATCH_MARKER, False):
        return

    @wraps(original)
    def fit_reconstruction_latent_classifier(*, config=None, **kwargs: Any):
        if config is not None:
            config = _normalize_config(reconstruction_encoder, config)
        return original(config=config, **kwargs)

    setattr(fit_reconstruction_latent_classifier, _CLASSIFIER_CONFIG_PATCH_MARKER, True)
    reconstruction_encoder.fit_reconstruction_latent_classifier = fit_reconstruction_latent_classifier


def install() -> None:
    """Patch user-facing reconstruction config parsing."""

    reconstruction_encoder = importlib.import_module("neureptrace.decoding.reconstruction_encoder")
    if getattr(reconstruction_encoder, _PATCH_MARKER, False):
        return

    _install_hidden_units_normalizer(reconstruction_encoder)
    _install_fit_scope_normalizer(reconstruction_encoder)
    _install_classifier_config_normalizer(reconstruction_encoder)
    original = reconstruction_encoder.reconstruction_encoder_config

    @wraps(original)
    def reconstruction_encoder_config(*, standardize: bool | int | str = False, **kwargs: Any):
        return original(standardize=_normalize_bool(standardize, name="standardize"), **kwargs)

    reconstruction_encoder.reconstruction_encoder_config = reconstruction_encoder_config
    setattr(reconstruction_encoder, _PATCH_MARKER, True)


__all__ = ["install"]
