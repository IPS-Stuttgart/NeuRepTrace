from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.loso_time_decode import _feasible_source_cv_splits, _preprocessed_data_for_outer_fold


def test_loso_preprocessing_rejects_missing_group_labels() -> None:
    data = np.zeros((3, 1, 2), dtype=float)
    times = np.array([0.0, 0.01])
    metadata = pd.DataFrame({"subject": ["s1", None, "s2"]})

    with pytest.raises(ValueError, match="LOSO group column 'subject'.*missing values"):
        _preprocessed_data_for_outer_fold(
            data,
            times,
            metadata,
            normalization="none",
            normalization_scope="global",
            baseline_window=(0.0, 0.01),
            train_indices=np.array([0, 1]),
            group_column="subject",
        )


def test_loso_source_cv_rejects_missing_group_labels() -> None:
    labels = np.array([0, 1, 0, 1])
    groups = np.array(["s1", "s1", None, "s2"], dtype=object)

    with pytest.raises(ValueError, match="Source-window selection groups.*missing values"):
        list(_feasible_source_cv_splits(labels, groups, 2))
