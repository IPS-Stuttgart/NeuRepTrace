from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_ecdf_uniform import (
    SOURCE_ECDF_CATEGORY,
    SourceEcdfConfig,
    apply_source_ecdf_transform,
    fit_source_ecdf_map,
    fit_source_ecdf_transform,
    source_ecdf_config,
)


def test_source_ecdf_transform_uses_source_quantiles_only() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    rows = np.asarray([[-1.0], [0.0], [1.5], [4.0]], dtype=float)

    result = fit_source_ecdf_transform(source_features=source, test_features=rows, config={"n_quantiles": 4})

    assert result.metadata["source_ecdf_protocol_category"] == SOURCE_ECDF_CATEGORY
    assert result.metadata["source_ecdf_uses_test_features_for_fitting"] is False
    assert result.metadata["source_ecdf_uses_test_labels"] is False
    assert np.all(result.test_features > 0.0)
    assert np.all(result.test_features < 1.0)
    assert np.allclose(result.train_features.ravel(), np.asarray([1e-6, 1.0 / 3.0, 2.0 / 3.0, 1.0 - 1e-6], dtype=np.float32), atol=1e-6)


def test_source_ecdf_accepts_one_pass_feature_iterables() -> None:
    source_rows = ([float(value)] for value in range(4))
    test_rows = ([1.5], [4.0])

    result = fit_source_ecdf_transform(source_features=source_rows, test_features=(row for row in test_rows), config={"n_quantiles": 4})

    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (2, 1)
    assert np.allclose(result.test_features.ravel(), [0.5, 1.0 - 1e-6])


def test_source_ecdf_constant_feature_maps_to_half() -> None:
    ecdf_map = fit_source_ecdf_map([[2.0], [2.0], [2.0]])
    transformed = apply_source_ecdf_transform([[1.0], [2.0], [3.0]], ecdf_map)

    assert np.allclose(transformed, 0.5)


def test_source_ecdf_apply_accepts_one_pass_iterables() -> None:
    ecdf_map = fit_source_ecdf_map(([float(value)] for value in range(4)), config={"n_quantiles": 4})

    transformed = apply_source_ecdf_transform(([1.5], [4.0]), ecdf_map)

    assert transformed.shape == (2, 1)
    assert np.allclose(transformed.ravel(), [0.5, 1.0 - 1e-6])


def test_source_ecdf_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_ecdf_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_ecdf_config_validation() -> None:
    assert source_ecdf_config(n_quantiles="3").n_quantiles == 3
    with pytest.raises(ValueError, match="n_quantiles"):
        source_ecdf_config(n_quantiles=0)
    with pytest.raises(ValueError, match="epsilon"):
        source_ecdf_config(epsilon=0.5)


def test_source_ecdf_config_dataclass_validation() -> None:
    config = SourceEcdfConfig(n_quantiles="3", epsilon="1e-4")  # type: ignore[arg-type]

    assert config.n_quantiles == 3
    assert config.epsilon == 1e-4

    with pytest.raises(ValueError, match="n_quantiles"):
        SourceEcdfConfig(n_quantiles=0)
    with pytest.raises(ValueError, match="n_quantiles"):
        SourceEcdfConfig(n_quantiles=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="epsilon"):
        SourceEcdfConfig(epsilon=0.5)


def test_source_ecdf_transform_accepts_direct_dataclass_config() -> None:
    result = fit_source_ecdf_transform(
        source_features=[[0.0], [1.0], [2.0]],
        test_features=[[0.5]],
        config=SourceEcdfConfig(n_quantiles="2", epsilon="1e-4"),  # type: ignore[arg-type]
    )

    assert result.metadata["source_ecdf_requested_quantiles"] == 2
    assert result.metadata["source_ecdf_epsilon"] == 1e-4
