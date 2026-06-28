from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace import openneuro_real_shuffle_report as report


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 1, np.int64(1), 1.0, np.float64(1.0), "1", "1.0", "1.00", "true", "t", "yes", "y", "on"],
)
def test_real_shuffle_bool_token_accepts_true_csv_tokens(value):
    assert report._as_bool_token(value) is True


@pytest.mark.parametrize(
    "value",
    [False, np.bool_(False), 0, np.int64(0), 0.0, np.float64(0.0), "0", "0.0", "0.00", "false", "f", "no", "n", "off", None, pd.NA, np.nan, ""],
)
def test_real_shuffle_bool_token_accepts_false_and_missing_csv_tokens(value):
    assert report._as_bool_token(value) is False


@pytest.mark.parametrize("value", ["maybe", 2, np.int64(2), 0.5, np.float64(0.5), np.asarray([1])])
def test_real_shuffle_bool_token_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="boolean provenance"):
        report._as_bool_token(value)


def test_validate_shuffle_provenance_accepts_numeric_csv_booleans():
    real = {
        "quality": pd.DataFrame(
            {
                "label_shuffle_control": [0.0],
                "label_shuffle_seed": [np.nan],
            }
        )
    }
    shuffle = {
        "quality": pd.DataFrame(
            {
                "label_shuffle_control": [1.0],
                "label_shuffle_seed": [17.0],
            }
        )
    }

    provenance = report._validate_shuffle_provenance(real, shuffle)

    assert provenance["real_label_shuffle_control"] is False
    assert provenance["shuffle_label_shuffle_control"] is True
    assert provenance["shuffle_label_shuffle_seed"] == "17.0"
