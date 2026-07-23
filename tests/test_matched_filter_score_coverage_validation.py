from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.matched_filter_detection import score_stimulus_event_templates


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.1])
def test_score_templates_rejects_invalid_minimum_coverage(coverage: float):
    observations = pd.DataFrame({"time": [0.0]})
    templates = pd.DataFrame()

    with pytest.raises(ValueError, match=r"min_template_coverage must be in \(0, 1\]"):
        score_stimulus_event_templates(
            observations,
            templates,
            min_template_coverage=coverage,
        )
