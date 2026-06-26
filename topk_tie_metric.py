import numpy as np


def top_k_accuracy(probabilities, labels, k):
    matrix = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if matrix.size == 0:
        return float('nan')
    k = min(max(int(k), 1), matrix.shape[1])
    thresholds = np.partition(matrix, -k, axis=1)[:, -k]
    return float(np.mean([row[label] >= threshold for row, label, threshold in zip(matrix, labels, thresholds, strict=True)]))
