"""Validate Riemannian transfer label/domain vectors before masking or fitting."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_riemannian_vector_validation_patch_installed"


def _flat_object_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    """Return a 1-D object vector while rejecting true matrix-shaped inputs."""

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            vector = np.empty(1, dtype=object)
            vector[0] = array.item()
        elif array.ndim == 1:
            vector = array.reshape(-1)
        elif array.ndim == 2 and 1 in array.shape:
            vector = array.reshape(-1)
        else:
            raise ValueError(f"{name} must be one-dimensional or a single-row/single-column vector; got shape {array.shape}.")
    else:
        if isinstance(values, (str, bytes)):
            items = [values]
        else:
            try:
                items = list(values)
            except TypeError:
                items = [values]
        vector = np.empty(len(items), dtype=object)
        for index, item in enumerate(items):
            vector[index] = item

    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match source_covariances rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _domain_ids(n_rows: int, source_domains: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    if source_domains is None:
        return np.zeros(n_rows, dtype=int)
    domains = _flat_object_vector(source_domains, expected_length=n_rows, name="source_domains")
    for domain in domains.tolist():
        try:
            hash(domain)
        except TypeError as exc:
            raise ValueError(f"source_domains must contain hashable values; got {domain!r}.") from exc
    return domains


def _label_vector(source_labels: Sequence[Any] | np.ndarray, *, expected_length: int) -> np.ndarray:
    return _flat_object_vector(source_labels, expected_length=expected_length, name="source_labels")


def install() -> None:
    """Patch Riemannian transfer vector validation."""

    riemannian = importlib.import_module("neureptrace.decoding.riemannian")
    original_fit_predict_transfer = riemannian.fit_predict_riemannian_transfer
    if getattr(original_fit_predict_transfer, _PATCH_MARKER, False):
        return
    original_fit_predict_procrustes = riemannian.fit_predict_riemannian_procrustes

    @wraps(original_fit_predict_transfer)
    def fit_predict_riemannian_transfer(
        source_covariances,
        source_labels,
        target_covariances,
        *,
        source_domains=None,
        estimator=None,
        tangent_reference_scope="source",
        epsilon=riemannian.DEFAULT_RIEMANNIAN_EPSILON,
    ):
        transfer = riemannian.riemannian_tangent_transfer_features(
            source_covariances,
            target_covariances,
            source_domains=source_domains,
            tangent_reference_scope=tangent_reference_scope,
            epsilon=epsilon,
        )
        labels = _label_vector(source_labels, expected_length=transfer.source_features.shape[0])
        classifier = riemannian._default_estimator() if estimator is None else riemannian.clone(estimator)
        classifier.fit(transfer.source_features, labels)
        predictions = np.asarray(classifier.predict(transfer.target_features))
        return classifier, transfer, predictions

    @wraps(original_fit_predict_procrustes)
    def fit_predict_riemannian_procrustes(
        source_covariances,
        source_labels,
        target_covariances,
        *,
        source_domains=None,
        source_anchor_covariances_by_domain=None,
        target_anchor_covariances=None,
        rotation_mode="none",
        estimator=None,
        tangent_reference_scope="source",
        epsilon=riemannian.DEFAULT_RIEMANNIAN_EPSILON,
    ):
        transfer = riemannian.riemannian_procrustes_transfer_features(
            source_covariances,
            target_covariances,
            source_domains=source_domains,
            source_anchor_covariances_by_domain=source_anchor_covariances_by_domain,
            target_anchor_covariances=target_anchor_covariances,
            rotation_mode=rotation_mode,
            tangent_reference_scope=tangent_reference_scope,
            epsilon=epsilon,
        )
        labels = _label_vector(source_labels, expected_length=transfer.source_features.shape[0])
        classifier = riemannian._default_estimator() if estimator is None else riemannian.clone(estimator)
        classifier.fit(transfer.source_features, labels)
        predictions = np.asarray(classifier.predict(transfer.target_features))
        return classifier, transfer, predictions

    setattr(fit_predict_riemannian_transfer, _PATCH_MARKER, True)
    setattr(fit_predict_riemannian_procrustes, _PATCH_MARKER, True)
    riemannian._domain_ids = _domain_ids
    riemannian.fit_predict_riemannian_transfer = fit_predict_riemannian_transfer
    riemannian.fit_predict_riemannian_procrustes = fit_predict_riemannian_procrustes


__all__ = ["install"]
