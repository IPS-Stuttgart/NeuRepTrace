from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neureptrace.decoding._jda_math import source_weights
from neureptrace.decoding._jda_validate import component_count, encode_classes, feature_matrix, object_vector


@dataclass(frozen=True, slots=True)
class PreparedJDA:
    source: np.ndarray
    target: np.ndarray
    standardized: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    classes: tuple[Any, ...]
    labels: np.ndarray
    weights: np.ndarray
    centering: np.ndarray
    components: int


def prepare_jda(source_features, source_labels, target_features, *, n_components, balance_source_classes, standardize):
    source = feature_matrix(source_features, "source_features")
    target = feature_matrix(target_features, "target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have equal width")
    raw_labels = object_vector(source_labels, source.shape[0])
    classes, labels = encode_classes(raw_labels)
    joint = np.vstack([source, target])
    mean = joint.mean(axis=0) if standardize else np.zeros(joint.shape[1])
    centered = joint - mean
    scale = centered.std(axis=0, ddof=1) if standardize and joint.shape[0] > 1 else np.ones(joint.shape[1])
    scale = np.maximum(scale, 1e-12)
    standardized = centered / scale
    centering = np.eye(joint.shape[0]) - np.full((joint.shape[0], joint.shape[0]), 1.0 / joint.shape[0])
    components = component_count(n_components, joint.shape[0], joint.shape[1])
    weights = source_weights(labels, len(classes), balance_source_classes)
    return PreparedJDA(source, target, standardized, mean, scale, classes, labels, weights, centering, components)
