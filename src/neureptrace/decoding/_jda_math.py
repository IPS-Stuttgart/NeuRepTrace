from __future__ import annotations

import numpy as np

_MIN_SCALE = 1e-12


def discrepancy_matrix(ns, nt, source_labels, target_labels, n_classes, source_weights, conditional_weight):
    target_weights = np.full(nt, 1.0 / nt)
    vector = np.concatenate([source_weights, -target_weights])
    matrix = np.outer(vector, vector)
    for class_index in range(n_classes):
        source_mask = source_labels == class_index
        target_mask = target_labels == class_index
        if not np.any(target_mask):
            continue
        class_vector = np.zeros(ns + nt)
        class_vector[:ns][source_mask] = 1.0 / np.count_nonzero(source_mask)
        class_vector[ns:][target_mask] = -1.0 / np.count_nonzero(target_mask)
        matrix += conditional_weight * np.outer(class_vector, class_vector)
    norm = float(np.linalg.norm(matrix))
    return matrix / norm if norm > _MIN_SCALE else matrix


def centroid_probabilities(source, labels, target, n_classes, temperature):
    centroids = np.vstack([source[labels == index].mean(axis=0) for index in range(n_classes)])
    distances = ((target[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    logits = -distances / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(logits, -50.0, 50.0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, probabilities.argmax(axis=1).astype(int)


def source_weights(labels, n_classes, balance):
    if not balance:
        return np.full(labels.shape[0], 1.0 / labels.shape[0])
    weights = np.zeros(labels.shape[0])
    for index in range(n_classes):
        mask = labels == index
        weights[mask] = 1.0 / (n_classes * np.count_nonzero(mask))
    return weights


def canonical_projection(matrix):
    output = np.asarray(matrix, dtype=float).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1
    return output
