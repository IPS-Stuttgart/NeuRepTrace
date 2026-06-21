from neureptrace.decoding import source_alignment, source_alignment_core


def test_source_alignment_core_reexports_canonical_metadata():
    config = source_alignment_core.source_alignment_config(
        method="mcca",
        target_projection="pseudo_label_target_calibrated_alignment",
    )
    metadata = config.static_metadata()

    assert source_alignment_core.SourceAlignmentConfig is source_alignment.SourceAlignmentConfig
    assert metadata["alignment_pseudo_label_target_calibrated"] is True
    assert metadata["alignment_uses_unlabeled_target_data"] is True
    assert metadata["alignment_valid_for_benchmark"] is False
    assert metadata["alignment_valid_for_strict_source_only"] is False
