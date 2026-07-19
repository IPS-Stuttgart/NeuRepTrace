from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.paired_stats import paired_decoder_statistics


def test_paired_decoder_statistics_rejects_duplicate_metric_names() -> None:
    subject_metrics = pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "subject": ["s1", "s2", "s1", "s2"],
            "effect_accuracy": [0.8, 0.7, 0.6, 0.5],
        }
    )

    with pytest.raises(ValueError, match=r"duplicate names.*effect_accuracy"):
        paired_decoder_statistics(
            subject_metrics,
            metrics=("effect_accuracy", "effect_accuracy"),
        )
