from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.paired_stats import paired_decoder_statistics


def test_paired_statistics_reject_subject_ids_that_collapse_to_the_same_string() -> None:
    subject_metrics = pd.DataFrame(
        [
            {"decoder": "lda", "subject": 1, "effect_accuracy": 0.70},
            {"decoder": "lda", "subject": 2, "effect_accuracy": 0.65},
            {"decoder": "logistic", "subject": "1", "effect_accuracy": 0.60},
            {"decoder": "logistic", "subject": "2", "effect_accuracy": 0.55},
        ]
    )

    with pytest.raises(ValueError, match="ambiguous subject identifiers"):
        paired_decoder_statistics(
            subject_metrics,
            metrics=("effect_accuracy",),
            n_permutations=4,
        )
