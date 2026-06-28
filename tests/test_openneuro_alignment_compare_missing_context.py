from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from neureptrace.openneuro_alignment_compare import build_target_calibrated_comparison


def _variant_row(
    *,
    artifact_name: str,
    alignment_method: str,
    alignment_anchor_mode: str,
    alignment_target_projection: str,
    alignment_valid_for_benchmark: bool,
    selection_score: float,
) -> dict[str, object]:
    return {
        "dataset": "ds006629",
        "mode": "full",
        "subjects": "1-6",
        "runs": "all",
        "label_shuffle_control": False,
        "label_shuffle_seed": "",
        "selection_metric": "balanced_accuracy",
        "artifact_name": artifact_name,
        "alignment_method": alignment_method,
        "alignment_anchor_mode": alignment_anchor_mode,
        "alignment_target_projection": alignment_target_projection,
        "alignment_valid_for_benchmark": alignment_valid_for_benchmark,
        "selection_score": selection_score,
        "selection_value": selection_score,
    }


def test_target_calibrated_comparison_matches_raw_after_csv_roundtrip() -> None:
    variants = pd.DataFrame(
        [
            _variant_row(
                artifact_name="raw",
                alignment_method="none",
                alignment_anchor_mode="class_mean",
                alignment_target_projection="group_projection",
                alignment_valid_for_benchmark=True,
                selection_score=0.40,
            ),
            _variant_row(
                artifact_name="strict-alignment",
                alignment_method="mcca",
                alignment_anchor_mode="class_repetition",
                alignment_target_projection="group_projection",
                alignment_valid_for_benchmark=True,
                selection_score=0.50,
            ),
            _variant_row(
                artifact_name="target-calibrated-alignment",
                alignment_method="mcca",
                alignment_anchor_mode="class_repetition",
                alignment_target_projection="target_calibrated_alignment",
                alignment_valid_for_benchmark=False,
                selection_score=0.62,
            ),
        ]
    )

    reloaded = pd.read_csv(StringIO(variants.to_csv(index=False)))
    assert pd.isna(reloaded.loc[0, "label_shuffle_seed"])

    comparison = build_target_calibrated_comparison(reloaded, min_delta=0.01)

    assert comparison.loc[0, "decision"] == "target_calibrated_beats_strict_source_only"
    assert comparison.loc[0, "strict_artifact"] == "strict-alignment"
    assert comparison.loc[0, "raw_artifact"] == "raw"
    assert comparison.loc[0, "score_delta_target_calibrated_minus_strict"] == pytest.approx(0.12)
    assert comparison.loc[0, "score_delta_target_calibrated_minus_raw"] == pytest.approx(0.22)
