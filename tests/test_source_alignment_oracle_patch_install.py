from __future__ import annotations


def test_source_alignment_oracle_patch_is_installed_on_package_import():
    import neureptrace  # noqa: F401
    from neureptrace.decoding import source_alignment

    config = source_alignment.SourceAlignmentConfig(
        method="procrustes",
        target_projection=source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    )

    metadata = config.static_metadata()

    assert metadata["alignment_oracle_target_calibrated"] is True
    assert metadata["alignment_debug_upper_bound"] is True
    assert metadata["alignment_valid_for_benchmark"] is False
    assert metadata["alignment_valid_for_strict_source_only"] is False
    assert metadata["alignment_oracle_target_projection_source"] == "scored_heldout_target_rows"
    assert metadata["alignment_protocol_note"].startswith("uses scored held-out target labels or anchors")
