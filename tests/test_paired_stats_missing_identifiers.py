from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.paired_stats import paired_decoder_statistics


def _subject_metrics() -> pd.DataFrame:
    rows = []
    for subject in ("sub-01", "sub-02"):
        rows.extend(
            [
                {"decoder": "lda", "subject": subject, "effect_accuracy": 0.5},
                {"decoder": "logistic", "subject": subject, "effect_accuracy": 0.7},
            ]
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("identifier", ["decoder", "subject"])
def test_paired_statistics_rejects_missing_pairing_identifiers(identifier: str) -> None:
    subject_metrics = _subject_metrics()
    if identifier == "decoder":
        subject_metrics.loc[subject_metrics["decoder"].eq("lda"), identifier] = np.nan
    else:
        subject_metrics.loc[subject_metrics["subject"].eq("sub-02"), identifier] = np.nan

    with pytest.raises(ValueError, match="missing decoder or subject identifiers"):
        paired_decoder_statistics(
            subject_metrics,
            metrics=("effect_accuracy",),
        )
