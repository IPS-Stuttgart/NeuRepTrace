from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

JDA_PROTOCOL = "unlabeled_target_joint_distribution_adaptation"
JDA_CATEGORY = "2_unlabeled_target_adaptive"


@dataclass(frozen=True, slots=True)
class JointDistributionResult:
    source_features: np.ndarray
    target_features: np.ndarray
    projection: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    classes: tuple[Any, ...]
    target_assignments: np.ndarray
    target_probabilities: np.ndarray
    eigenvalues: np.ndarray
    n_iterations: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)
