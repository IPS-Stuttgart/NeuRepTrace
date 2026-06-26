from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401 - installs runtime patches
from neureptrace import mne_time_decode
from neureptrace.decoding.source_alignment import (
    PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    SourceAlignmentResult,
    source_alignment_config,
)


def test_pseudo_label_alignment_fallback_is_not_marked_benchmark_valid() -> None:
    source_only_result = SourceAlignmentResult(
        train_features=np.ones((4, 2), dtype=float),
        test_features=np.ones((2, 2), dtype=float),
        metadata={
            "alignment_target_projection": "group_projection",
            "alignment_target_projection_fit": "source_group_projection",
            "alignment_valid_for_benchmark": True,
            "alignment_valid_for_strict_source_only": True,
        },
        diagnostics={
            "alignment_target_projection": "group_projection",
            "alignment_target_projection_fit": "source_group_projection",
            "alignment_valid_for_benchmark": True,
            "alignment_valid_for_strict_source_only": True,
        },
    )
    config = source_alignment_config(
        method="procrustes",
        target_projection=PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    )

    fallback = mne_time_decode._pseudo_label_target_alignment_fallback_result(
        source_only_result,
        config,
        stop_reason="alignment_missing_pseudo_anchors",
    )

    assert fallback.metadata["alignment_target_projection"] == PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT
    assert fallback.diagnostics["alignment_target_projection"] == PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT
    assert fallback.metadata["alignment_target_projection_fit"] == "source_group_projection_fallback"
    assert fallback.diagnostics["alignment_target_projection_fit"] == "source_group_projection_fallback"
    assert fallback.metadata["alignment_valid_for_benchmark"] is False
    assert fallback.diagnostics["alignment_valid_for_benchmark"] is False
    assert fallback.metadata["alignment_valid_for_strict_source_only"] is False
    assert fallback.diagnostics["alignment_valid_for_strict_source_only"] is False
    assert fallback.metadata["alignment_pseudo_label_fallback"] is True
    assert fallback.diagnostics["alignment_pseudo_label_fallback"] is True
    assert fallback.metadata["alignment_pseudo_label_fallback_reason"] == "alignment_missing_pseudo_anchors"
    assert fallback.diagnostics["uses_unlabeled_target_data"] is True
    assert "category-2" in fallback.diagnostics["alignment_protocol_note"]
