from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.decoding.source_prior import estimate_source_class_prior


def test_source_prior_keeps_numpy_nat_kinds_as_distinct_classes() -> None:
    datetime_nat = np.datetime64("NaT")
    timedelta_nat = np.timedelta64("NaT")

    prior, classes = estimate_source_class_prior(
        [datetime_nat, timedelta_nat, np.datetime64("NaT")]
    )

    assert classes.shape == (2,)
    assert isinstance(classes[0], np.datetime64)
    assert isinstance(classes[1], np.timedelta64)
    assert np.isnat(classes[0])
    assert np.isnat(classes[1])
    np.testing.assert_allclose(prior, [2.0 / 3.0, 1.0 / 3.0])


def test_source_prior_keeps_pandas_na_separate_from_float_nan() -> None:
    prior, classes = estimate_source_class_prior(
        [float("nan"), pd.NA, np.float64("nan")]
    )

    assert classes.shape == (2,)
    assert np.isnan(classes[0])
    assert classes[1] is pd.NA
    np.testing.assert_allclose(prior, [2.0 / 3.0, 1.0 / 3.0])
