from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.bushmeg_cue_source_weights import (
    CueSourceWeights,
    CueSubjectData,
    cue_subject_feature_vector,
    normalize_cue_feature_kinds,
    normalize_cue_source_weighting_mode,
)


def test_cue_source_weights_softmax_topk_returns_mean_one_weights():
    features = {
        "target": np.array([1.0, 0.0, 0.0]),
        "near": np.array([0.9, 0.1, 0.0]),
        "far": np.array([-1.0, 0.0, 0.0]),
        "mid": np.array([0.5, 0.5, 0.0]),
    }
    cue_weights = CueSourceWeights(
        features,
        mode="softmax_top_k",
        top_k=2,
        temperature=0.25,
        blend=1.0,
    )

    weights = cue_weights.for_fold("target", ["near", "far", "mid"])

    assert set(weights) == {"near", "far", "mid"}
    assert np.isclose(np.mean(list(weights.values())), 1.0)
    assert weights["near"] > weights["mid"] > weights["far"]
    assert weights["far"] == 0.0


def test_cue_source_weights_blend_keeps_nonnearest_sources_alive():
    cue_weights = CueSourceWeights(
        {
            "target": np.array([1.0, 0.0]),
            "near": np.array([1.0, 0.1]),
            "far": np.array([-1.0, 0.0]),
        },
        mode="top_k",
        top_k=1,
        blend=0.5,
    )

    weights = cue_weights.for_fold("target", ["near", "far"])

    assert np.isclose(np.mean(list(weights.values())), 1.0)
    assert weights["near"] > weights["far"] > 0.0


def test_cue_feature_aliases_are_normalized():
    assert normalize_cue_source_weighting_mode("softmax-topk") == "softmax_top_k"
    assert normalize_cue_feature_kinds(["gfp", "baseline-var", "evoked"]) == (
        "evoked_gfp",
        "baseline_logvar",
        "evoked_mean",
    )


def test_cue_subject_feature_vector_concatenates_requested_parts():
    data = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]],
        ],
        dtype=np.float32,
    )
    subject = CueSubjectData(
        subject="s1",
        data=data,
        times=np.array([-0.10, -0.05, 0.10, 0.20]),
        metadata=pd.DataFrame(),
    )

    vector = cue_subject_feature_vector(
        subject,
        feature_kinds=["baseline_logvar", "evoked_gfp", "evoked_mean"],
        baseline_window=(-0.10, -0.05),
        response_window=(0.10, 0.20),
        temporal_bins=2,
    )

    assert vector.shape == (2 + 2 + 4,)
    assert np.all(np.isfinite(vector))
