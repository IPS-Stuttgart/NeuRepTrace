from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding.source_zca import apply_source_zca_transform, fit_source_zca_reference, fit_source_zca_transform


def test_source_zca_accepts_dataframe_feature_matrices() -> None:
    source = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        columns=["sensor_a", "sensor_b"],
    )
    test = pd.DataFrame(
        [[0.5, 0.5], [2.0, 0.0]],
        columns=["sensor_a", "sensor_b"],
    )

    dataframe_result = fit_source_zca_transform(source_features=source, test_features=test)
    array_result = fit_source_zca_transform(
        source_features=source.to_numpy(),
        test_features=test.to_numpy(),
    )
    reference = fit_source_zca_reference(source)
    reapplied = apply_source_zca_transform(test, reference)

    np.testing.assert_allclose(dataframe_result.train_features, array_result.train_features)
    np.testing.assert_allclose(dataframe_result.test_features, array_result.test_features)
    np.testing.assert_allclose(reapplied, array_result.test_features)
    assert dataframe_result.metadata["source_zca_feature_dim"] == 2


@pytest.mark.parametrize(
    ("bad_features", "message"),
    [
        (pd.DataFrame([[True, False], [False, True]], columns=["sensor_a", "sensor_b"]), "numeric feature values"),
        (pd.DataFrame([[1.0 + 2.0j, 0.0], [2.0, 1.0]], columns=["sensor_a", "sensor_b"]), "real-valued feature values"),
    ],
)
def test_source_zca_validates_dataframe_contents(bad_features: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_source_zca_transform(
            source_features=bad_features,
            test_features=[[0.5, 0.5]],
        )
