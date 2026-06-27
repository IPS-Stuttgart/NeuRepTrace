from __future__ import annotations


def test_package_import_installs_source_alignment_target_offset_patch() -> None:
    import neureptrace  # noqa: F401
    from neureptrace.decoding import source_alignment

    assert getattr(
        source_alignment,
        "_neureptrace_source_alignment_target_calibration_offsets_patch_installed",
        False,
    )
