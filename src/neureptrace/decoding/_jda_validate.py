from __future__ import annotations

import numpy as np


def feature_matrix(values, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    return matrix


def object_vector(values, expected_length):
    items = list(values)
    if len(items) != expected_length:
        raise ValueError("source_labels must contain one value per row")
    output = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        output[index] = value
    return output


def encode_classes(labels):
    classes = tuple(dict.fromkeys(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("at least two source classes are required")
    for value in classes:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("source labels must be hashable") from exc
    lookup = {value: index for index, value in enumerate(classes)}
    encoded = np.asarray([lookup[value] for value in labels.tolist()], dtype=int)
    return classes, encoded


def component_count(value, n_samples, n_features):
    maximum = max(1, min(n_features, n_samples - 1))
    if isinstance(value, str) and value.strip().lower() in {"all", "full"}:
        return maximum
    return min(positive_int(value, "n_components"), maximum)


def positive_int(value, name):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 1 or parsed % 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


def positive_float(value, name):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def nonnegative_float(value, name):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed
