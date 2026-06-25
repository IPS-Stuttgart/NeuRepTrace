from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.kernel_mean_matching import (
    kernel_mean_matching_weights,
    kmm_config,
    normalize_kmm_epsilon,
    resolve_kmm_gamma,
)


def test_kmm_rejects_boolean_gamma_values() -> None:
    source = np.asarray([[0.0], [1.0]], dtype=float)
    target = np.asarray([[0.25], [0.75]], dtype=float)

    with pytest.raises(ValueError, match="gamma"):
        resolve_kmm_gamma(True, source, target)

    with pytest.raises(ValueError, match="gamma"):
        resolve_kmm_gamma(np.bool_(True), source, target)

    with pytest.raises(ValueError, match="gamma"):
        kernel_mean_matching_weights(source, target, gamma=True)

    with pytest.raises(ValueError, match="gamma"):
        kmm_config(gamma=np.bool_(True))


def test_kmm_rejects_boolean_epsilon_values() -> None:
    source = np.asarray([[0.0], [1.0]], dtype=float)
    target = np.asarray([[0.25], [0.75]], dtype=float)

    with pytest.raises(ValueError, match="epsilon"):
        normalize_kmm_epsilon(False, n_source=source.shape[0])

    with pytest.raises(ValueError, match="epsilon"):
        normalize_kmm_epsilon(np.bool_(False), n_source=source.shape[0])

    with pytest.raises(ValueError, match="epsilon"):
        kernel_mean_matching_weights(source, target, epsilon=False)

    with pytest.raises(ValueError, match="epsilon"):
        kmm_config(epsilon=np.bool_(False))
