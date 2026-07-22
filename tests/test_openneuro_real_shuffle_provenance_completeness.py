from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace import openneuro_real_shuffle_report as report


def test_validate_shuffle_provenance_rejects_partially_missing_control_flags() -> None:
    real = {
        "quality": pd.DataFrame(
            {
                "label_shuffle_control": [0.0, np.nan],
            }
        )
    }
    shuffle = {
        "quality": pd.DataFrame(
            {
                "label_shuffle_control": [1.0, 1.0],
            }
        )
    }

    with pytest.raises(ValueError, match="missing 'label_shuffle_control' provenance"):
        report._validate_shuffle_provenance(real, shuffle)


def test_validate_matched_provenance_rejects_partially_missing_protocol_values() -> None:
    real = {
        "observations": pd.DataFrame(
            {
                "decoder": ["logistic", np.nan],
            }
        )
    }
    shuffle = {
        "observations": pd.DataFrame(
            {
                "decoder": ["logistic", "logistic"],
            }
        )
    }

    with pytest.raises(ValueError, match="missing 'decoder' provenance"):
        report._validate_matched_provenance(real, shuffle)
