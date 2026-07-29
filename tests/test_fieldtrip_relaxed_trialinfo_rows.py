from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.io.fieldtrip_mat import MetadataColumnSpec, _trialinfo_to_frame


def test_relaxed_trialinfo_row_validation_pads_missing_rows():
    metadata = _trialinfo_to_frame(
        np.array([[1, 10], [2, 20]], dtype=int),
        n_trials=4,
        columns=(
            MetadataColumnSpec(name="stimulus_class", index=0),
            MetadataColumnSpec(name="condition", index=1),
        ),
        require_rows_equal_trials=False,
    )

    assert len(metadata) == 4
    assert metadata.loc[:1, "stimulus_class"].tolist() == [1, 2]
    assert metadata.loc[:1, "condition"].tolist() == [10, 20]
    assert metadata.loc[2:, ["stimulus_class", "condition"]].isna().all().all()


def test_relaxed_trialinfo_row_validation_keeps_trial_column_aligned():
    metadata = _trialinfo_to_frame(
        np.array([7], dtype=int),
        n_trials=3,
        columns=(),
        require_rows_equal_trials=False,
    )

    assert metadata["trial"].tolist() == [0, 1, 2]
    assert metadata.loc[0, "trialinfo_0"] == 7
    assert pd.isna(metadata.loc[1, "trialinfo_0"])
    assert pd.isna(metadata.loc[2, "trialinfo_0"])


def test_strict_trialinfo_row_validation_still_rejects_missing_rows():
    with pytest.raises(ValueError, match="trialinfo has 1 rows but FieldTrip data contains 3 trials"):
        _trialinfo_to_frame(
            np.array([7], dtype=int),
            n_trials=3,
            columns=(),
            require_rows_equal_trials=True,
        )
