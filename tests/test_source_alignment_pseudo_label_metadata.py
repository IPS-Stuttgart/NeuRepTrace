from neureptrace.decoding.source_alignment import (
    PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    _target_alignment_matrix,
    source_alignment_config,
)


def test_pseudo_label_target_calibrated_alignment_is_not_benchmark_valid():
    metadata = source_alignment_config(
        method="mcca",
        target_projection=PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    ).static_metadata()

    assert metadata["alignment_pseudo_label_target_calibrated"] is True
    assert metadata["alignment_uses_unlabeled_target_data"] is True
    assert metadata["alignment_valid_for_benchmark"] is False
    assert metadata["alignment_valid_for_strict_source_only"] is False
    assert metadata["alignment_strict_source_only"] is False
    assert "transductive" in metadata["alignment_protocol_note"]


def test_source_alignment_wrapper_preserves_private_test_helpers():
    assert callable(_target_alignment_matrix)
