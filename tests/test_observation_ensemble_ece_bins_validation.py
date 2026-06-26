import pandas as pd
import pytest

from neureptrace.observation_ensemble import summarize_ensemble_metrics


def test_summarize_ensemble_metrics_rejects_bool_ece_bins() -> None:
    observations = pd.DataFrame(
        {
            "true_label": [0, 1],
            "class_0": ["zero", "zero"],
            "class_1": ["one", "one"],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    )

    with pytest.raises(ValueError, match="ece_bins must be a positive integer"):
        summarize_ensemble_metrics(observations, ece_bins=True)
