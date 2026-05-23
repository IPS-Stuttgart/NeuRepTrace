from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_cue_source_weighting import (
    cue_source_weights_from_summaries,
    cue_window_tuple,
    normalize_cue_source_weighting,
)


def test_cue_source_weights_prefer_cue_similar_sources():
    summaries = {
        "target": np.array([1.0, 0.0, 0.0]),
        "near": np.array([0.9, 0.1, 0.0]),
        "far": np.array([-1.0, 0.0, 0.0]),
    }

    weights = cue_source_weights_from_summaries(
        summaries,
        test_subject="target",
        train_subjects=["near", "far"],
        mode="cue-evoked",
        temperature=0.25,
    )

    assert normalize_cue_source_weighting("cue-hybrid") == "cue_hybrid_correlation"
    assert weights is not None
    assert np.isclose(np.mean(list(weights.values())), 1.0)
    assert weights["near"] > weights["far"]


def test_cue_window_tuple_validates_configured_windows():
    assert cue_window_tuple([-0.1, 0.2], (-0.05, 0.25), key="cue_summary_window") == (-0.1, 0.2)
    assert cue_window_tuple("-0.1,0.2", (-0.05, 0.25), key="cue_summary_window") == (-0.1, 0.2)
