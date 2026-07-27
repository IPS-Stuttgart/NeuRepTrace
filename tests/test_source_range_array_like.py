from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding.source_range import source_feature_range, source_range_clip


def test_source_range_accepts_dataframe_feature_matrices() -> None:
    source = pd.DataFrame([[0.0, 10.0], [2.0, 12.0]], columns=["first", "second"])
    test = pd.DataFrame([[-5.0, 11.0], [5.0, 20.0]], columns=["first", "second"])

    lower, upper = source_feature_range(source)
    train, test_out, *_ = source_range_clip(source_features=source, test_features=test)

    np.testing.assert_allclose(lower, [0.0, 10.0])
    np.testing.assert_allclose(upper, [2.0, 12.0])
    np.testing.assert_allclose(train, source.to_numpy())
    np.testing.assert_allclose(test_out, [[0.0, 11.0], [2.0, 12.0]])


def test_source_range_rejects_boolean_dataframe_features() -> None:
    source = pd.DataFrame([[True, False], [False, True]])

    with pytest.raises(ValueError, match="source_features.*non-boolean"):
        source_feature_range(source)
