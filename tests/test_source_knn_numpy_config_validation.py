from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_knn import (
    SourceKNNConfig,
    fit_source_knn_reference,
    source_knn_config,
)


@pytest.mark.parametrize(
    "bad_k",
    [np.asarray(True), np.asarray(False), np.asarray([1])],
)
def test_source_knn_rejects_numpy_boolean_and_vector_k(bad_k) -> None:
    with pytest.raises(ValueError, match="k"):
        source_knn_config(k=bad_k)

    with pytest.raises(ValueError, match="k"):
        fit_source_knn_reference(
            source_features=[[0.0], [1.0]],
            source_labels=[0, 1],
            config=SourceKNNConfig(k=bad_k),
        )


@pytest.mark.parametrize(
    "bad_epsilon",
    [np.asarray(True), np.asarray(False), np.asarray([1.0])],
)
def test_source_knn_rejects_numpy_boolean_and_vector_epsilon(bad_epsilon) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        source_knn_config(epsilon=bad_epsilon)

    with pytest.raises(ValueError, match="epsilon"):
        fit_source_knn_reference(
            source_features=[[0.0], [1.0]],
            source_labels=[0, 1],
            config=SourceKNNConfig(epsilon=bad_epsilon),
        )


def test_source_knn_normalizes_scalar_numpy_numeric_config_values() -> None:
    cfg = source_knn_config(k=np.asarray(2), epsilon=np.asarray("1e-4"))

    assert cfg.k == 2
    assert cfg.epsilon == pytest.approx(1e-4)

    reference = fit_source_knn_reference(
        source_features=[[0.0], [1.0], [2.0]],
        source_labels=[0, 1, 1],
        config=SourceKNNConfig(k=np.asarray(2), epsilon=np.asarray("1e-4")),
    )

    assert reference.config.k == 2
    assert reference.config.epsilon == pytest.approx(1e-4)
