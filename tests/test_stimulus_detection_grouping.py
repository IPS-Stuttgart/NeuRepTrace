from __future__ import annotations

import numpy as np
import pandas as pd

from reptrace.stimulus_detection import fit_stimulus_detection_thresholds


THRESHOLD_WINDOW = (-0.30, -0.10)


def _variant_row(
    *,
    stream_id: str,
    feature_preprocessor: str,
    time: float,
    prob_a: float,
) -> dict:
    return {
        "subject": "sub-01",
        "stream_id": stream_id,
        "decoder": "logistic",
        "emission_mode": "calibrated",
        "feature_preprocessor": feature_preprocessor,
        # Keep an all-NaN default grouping column present to ensure pandas does
        # not silently drop rows when optional analysis metadata is unavailable.
        "pca_components": np.nan,
        "time": time,
        "window_start": time - 0.025,
        "window_stop": time + 0.025,
        "predicted_label": 0 if prob_a >= 0.5 else 1,
        "predicted_class": "A" if prob_a >= 0.5 else "B",
        "confidence": max(prob_a, 1.0 - prob_a),
        "class_0": "A",
        "class_1": "B",
        "prob_class_0": prob_a,
        "prob_class_1": 1.0 - prob_a,
    }


def test_default_stimulus_grouping_separates_analysis_variants() -> None:
    frame = pd.DataFrame(
        [
            _variant_row(stream_id="raw", feature_preprocessor="raw", time=-0.20, prob_a=0.20),
            _variant_row(stream_id="raw", feature_preprocessor="raw", time=0.10, prob_a=0.30),
            _variant_row(stream_id="z", feature_preprocessor="zscore", time=-0.20, prob_a=0.80),
            _variant_row(stream_id="z", feature_preprocessor="zscore", time=0.10, prob_a=0.85),
        ]
    )

    thresholds = fit_stimulus_detection_thresholds(
        frame,
        stream_columns=("stream_id",),
        threshold_window=THRESHOLD_WINDOW,
        threshold_quantile=1.0,
        target_classes=["A"],
    )

    by_variant = thresholds.set_index("feature_preprocessor")["score_threshold"].to_dict()

    assert set(by_variant) == {"raw", "zscore"}
    assert np.isclose(by_variant["raw"], 0.20)
    assert np.isclose(by_variant["zscore"], 0.80)
    assert thresholds["pca_components"].isna().all()
