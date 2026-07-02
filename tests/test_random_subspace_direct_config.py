from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.random_subspace import (
    RandomSubspaceEnsembleConfig,
    fit_random_subspace_ensemble,
)


def test_random_subspace_revalidates_direct_config_objects() -> None:
    config = RandomSubspaceEnsembleConfig(
        n_estimators="2",  # type: ignore[arg-type]
        feature_fraction="1.0",  # type: ignore[arg-type]
        min_features="1",  # type: ignore[arg-type]
        bootstrap_rows="false",  # type: ignore[arg-type]
        row_fraction="1.0",  # type: ignore[arg-type]
        random_state="7",  # type: ignore[arg-type]
        epsilon="1e-9",  # type: ignore[arg-type]
    )

    result = fit_random_subspace_ensemble(
        train_features=np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [1.0, 1.0],
                [1.1, 1.0],
                [1.0, 1.1],
            ],
            dtype=float,
        ),
        train_labels=["left", "left", "left", "right", "right", "right"],
        test_features=[[0.05, 0.05], [1.05, 1.05]],
        config=config,
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["random_subspace_n_estimators"] == 2
    assert result.metadata["random_subspace_bootstrap_rows"] is False
    assert result.metadata["random_subspace_row_fraction"] == 1.0
    assert result.metadata["random_subspace_random_state"] == 7


def test_random_subspace_rejects_invalid_direct_config_objects() -> None:
    config = RandomSubspaceEnsembleConfig(
        n_estimators=True,  # type: ignore[arg-type]
        feature_fraction=1.0,
        min_features=1,
        bootstrap_rows=False,
        row_fraction=1.0,
        random_state=7,
        epsilon=1e-9,
    )

    with pytest.raises(ValueError, match="n_estimators"):
        fit_random_subspace_ensemble(
            train_features=[[0.0], [1.0]],
            train_labels=[0, 1],
            test_features=[[0.5]],
            config=config,
        )
