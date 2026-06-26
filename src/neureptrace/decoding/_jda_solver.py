from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from neureptrace.decoding._jda_math import canonical_projection, centroid_probabilities, discrepancy_matrix


@dataclass(frozen=True, slots=True)
class SolvedJDA:
    source_latent: np.ndarray
    target_latent: np.ndarray
    projection: np.ndarray
    probabilities: np.ndarray
    assignments: np.ndarray
    eigenvalues: np.ndarray
    iterations: int
    converged: bool
    changes: tuple[float, ...]


def solve_jda(prepared, *, regularization, eigen_ridge, conditional_weight, max_iterations, temperature):
    ns = prepared.source.shape[0]
    nt = prepared.target.shape[0]
    source_z = prepared.standardized[:ns]
    target_z = prepared.standardized[ns:]
    probabilities, assignments = centroid_probabilities(source_z, prepared.labels, target_z, len(prepared.classes), temperature)
    projection = np.eye(prepared.standardized.shape[1], prepared.components)
    eigenvalues = np.zeros(prepared.components)
    source_latent = source_z @ projection
    target_latent = target_z @ projection
    changes = []
    converged = False
    run = 0
    for run in range(1, max_iterations + 1):
        discrepancy = discrepancy_matrix(ns, nt, prepared.labels, assignments, len(prepared.classes), prepared.weights, conditional_weight)
        width = prepared.standardized.shape[1]
        left = prepared.standardized.T @ discrepancy @ prepared.standardized + regularization * np.eye(width)
        right = prepared.standardized.T @ prepared.centering @ prepared.standardized + eigen_ridge * np.eye(width)
        values, vectors = eigh(left, right, check_finite=True)
        chosen = np.argsort(values)[: prepared.components]
        projection = canonical_projection(vectors[:, chosen])
        latent = prepared.standardized @ projection
        source_latent, target_latent = latent[:ns], latent[ns:]
        probabilities, refreshed = centroid_probabilities(source_latent, prepared.labels, target_latent, len(prepared.classes), temperature)
        change = float(np.mean(refreshed != assignments))
        changes.append(change)
        assignments = refreshed
        eigenvalues = np.asarray(values[chosen], dtype=float)
        if change == 0.0:
            converged = True
            break
    return SolvedJDA(source_latent, target_latent, projection, probabilities, assignments, eigenvalues, run, converged, tuple(changes))
