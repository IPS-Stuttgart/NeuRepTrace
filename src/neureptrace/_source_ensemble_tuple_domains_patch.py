"""Source-domain ensemble compatibility patches for composite domain IDs."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_ensemble_tuple_domains_patch_installed"


def _unique_domain_values(domains: np.ndarray, label_key) -> tuple[Hashable, ...]:
    unique: list[Hashable] = []
    seen: set[tuple[Any, ...]] = set()
    for domain in domains.tolist():
        key = label_key(domain)
        if key not in seen:
            seen.add(key)
            unique.append(domain)
    return tuple(unique)


def _domain_mask(domains: np.ndarray, domain: Hashable, label_key) -> np.ndarray:
    target_key = label_key(domain)
    return np.asarray([label_key(value) == target_key for value in domains.tolist()], dtype=bool)


def install() -> None:
    source_ensemble = importlib.import_module("neureptrace.decoding.source_ensemble")
    if getattr(source_ensemble, _PATCH_MARKER, False):
        return

    def domain_mask(domains: np.ndarray, domain: Hashable) -> np.ndarray:
        return _domain_mask(domains, domain, source_ensemble._label_key)

    def unique_domain_values(domains: np.ndarray) -> tuple[Hashable, ...]:
        return _unique_domain_values(domains, source_ensemble._label_key)

    def _domain_weights(
        mode: str,
        domain_probabilities: Mapping[Hashable, np.ndarray],
        source: np.ndarray,
        domains: np.ndarray,
        target: np.ndarray,
        *,
        temperature: float,
        epsilon: float,
    ) -> dict[Hashable, float]:
        domain_ids = tuple(domain_probabilities)
        if mode == "uniform":
            return {domain: 1.0 / len(domain_ids) for domain in domain_ids}
        if mode == "target_confidence":
            scores = np.asarray([np.mean(np.max(domain_probabilities[domain], axis=1)) for domain in domain_ids], dtype=float)
            return source_ensemble._softmax_scores(domain_ids, scores, temperature=temperature)
        if mode == "target_entropy":
            scores = []
            for domain in domain_ids:
                probabilities = np.maximum(domain_probabilities[domain], epsilon)
                scores.append(-float(np.mean(-np.sum(probabilities * np.log(probabilities), axis=1))))
            return source_ensemble._softmax_scores(domain_ids, np.asarray(scores, dtype=float), temperature=temperature)
        if mode == "target_feature_similarity":
            distances = np.asarray(
                [source_ensemble._feature_distribution_distance(source[domain_mask(domains, domain)], target) for domain in domain_ids],
                dtype=float,
            )
            return source_ensemble._softmax_scores(domain_ids, -distances, temperature=temperature)
        raise ValueError(f"Unhandled ensemble weighting mode {mode!r}.")

    # pylint: disable-next=too-many-arguments,too-many-locals
    def fit_source_domain_probability_ensemble(
        *,
        source_features: Sequence[Sequence[float]] | np.ndarray,
        source_labels: Sequence[Any] | np.ndarray,
        source_domains: Sequence[Hashable] | np.ndarray,
        target_features: Sequence[Sequence[float]] | np.ndarray,
        estimator=None,
        weighting: str = "uniform",
        temperature: float | str = source_ensemble.DEFAULT_TEMPERATURE,
        min_classes_per_domain: int | str = 2,
        epsilon: float | str = source_ensemble.DEFAULT_EPSILON,
    ):
        source = source_ensemble._feature_matrix(source_features, name="source_features")
        target = source_ensemble._feature_matrix(target_features, name="target_features")
        if source.shape[1] != target.shape[1]:
            raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")

        labels = source_ensemble._label_vector(source_labels, name="source_labels")
        if labels.shape[0] != source.shape[0]:
            raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {source.shape[0]}.")
        domains = source_ensemble._domain_vector(source_domains, expected_length=source.shape[0])
        domain_ids = unique_domain_values(domains)
        classes = source_ensemble._unique_label_vector(labels)
        if classes.shape[0] < 2:
            raise ValueError("At least two source classes are required.")

        class_indices = {source_ensemble._label_key(label): index for index, label in enumerate(classes.tolist())}
        encoded_labels = np.asarray([class_indices[source_ensemble._label_key(label)] for label in labels.tolist()], dtype=int)
        mode = source_ensemble.normalize_ensemble_weighting(weighting)
        temp = source_ensemble._positive_float(temperature, name="temperature")
        eps = source_ensemble._positive_float(epsilon, name="epsilon")
        min_classes = source_ensemble._positive_int(min_classes_per_domain, name="min_classes_per_domain")
        model_template = source_ensemble._default_estimator() if estimator is None else estimator

        models: dict[Hashable, source_ensemble.SourceDomainModel] = {}
        domain_probabilities: dict[Hashable, np.ndarray] = {}
        for domain in domain_ids:
            mask = domain_mask(domains, domain)
            domain_classes = source_ensemble._unique_label_vector(labels[mask])
            if domain_classes.shape[0] < min_classes:
                continue
            model = source_ensemble.clone(model_template)
            model.fit(source[mask], encoded_labels[mask])
            probabilities = source_ensemble._aligned_probabilities(model, target, classes=np.arange(classes.shape[0]), epsilon=eps)
            models[domain] = source_ensemble.SourceDomainModel(domain_id=domain, model=model, n_rows=int(np.sum(mask)), classes=domain_classes)
            domain_probabilities[domain] = probabilities
        if not models:
            raise ValueError("No source domain had enough classes to train a domain classifier.")

        weights = _domain_weights(mode, domain_probabilities, source, domains, target, temperature=temp, epsilon=eps)
        probabilities = np.zeros((target.shape[0], classes.shape[0]), dtype=float)
        for domain, domain_probability in domain_probabilities.items():
            probabilities += weights[domain] * domain_probability
        probabilities = source_ensemble._normalize_probability_rows(probabilities, epsilon=eps)
        predictions = classes[np.argmax(probabilities, axis=1)]
        return source_ensemble.SourceDomainEnsembleResult(
            probabilities=probabilities.astype(np.float32, copy=False),
            predictions=predictions,
            classes=classes,
            domain_weights=weights,
            domain_probabilities={domain: values.astype(np.float32, copy=False) for domain, values in domain_probabilities.items()},
            models=models,
            metadata=source_ensemble._metadata(
                mode=mode,
                n_source_rows=source.shape[0],
                n_target_rows=target.shape[0],
                feature_dim=source.shape[1],
                n_classes=classes.shape[0],
                n_source_domains=len(domain_ids),
                n_trained_domains=len(models),
                weights=weights,
                temperature=temp,
                min_classes=min_classes,
            ),
        )

    source_ensemble._unique_domain_values = unique_domain_values
    source_ensemble._domain_mask = domain_mask
    source_ensemble._domain_weights = _domain_weights
    source_ensemble.fit_source_domain_probability_ensemble = fit_source_domain_probability_ensemble
    setattr(source_ensemble, _PATCH_MARKER, True)


__all__ = ["install"]
