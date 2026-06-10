import pytest

from neureptrace.decoding.alignment_diagnostics import (
    AlignmentDiagnosticWarning,
    make_alignment_diagnostics,
    requested_component_count,
    warn_for_alignment_diagnostics,
)


def test_low_rank_class_mean_diagnostic_records_rank_cap():
    diagnostics = make_alignment_diagnostics(
        method="mcca",
        sample_mode="class_mean",
        n_alignment_rows=3,
        n_components=64,
        n_classes=3,
    )

    assert diagnostics.row_rank_cap == 2
    assert diagnostics.effective_components == 2
    assert diagnostics.is_low_rank_class_mean
    assert diagnostics.to_record()["is_low_rank_class_mean"] is True


def test_low_rank_class_mean_warns():
    diagnostics = make_alignment_diagnostics(
        method="hyperalignment",
        sample_mode="class_mean",
        n_alignment_rows=3,
        n_components=64,
        n_classes=3,
    )

    with pytest.warns(AlignmentDiagnosticWarning, match="severe bottleneck"):
        warn_for_alignment_diagnostics(diagnostics)


def test_group_projection_without_target_calibration_warns():
    diagnostics = make_alignment_diagnostics(
        method="mcca",
        sample_mode="class_repetition",
        n_alignment_rows=30,
        n_components=16,
        actual_components=16,
        target_projection_kind="group_projection",
        target_calibration_rows=0,
    )

    assert diagnostics.uses_group_projection_fallback
    assert not diagnostics.has_target_calibration
    with pytest.warns(AlignmentDiagnosticWarning, match="calibration-free fallback"):
        warn_for_alignment_diagnostics(diagnostics)


def test_target_calibrated_projection_does_not_warn_about_group_fallback():
    diagnostics = make_alignment_diagnostics(
        method="mcca",
        sample_mode="class_repetition",
        n_alignment_rows=30,
        n_components=16,
        actual_components=16,
        target_projection_kind="target_class_repetition",
        target_calibration_rows=30,
    )

    assert not diagnostics.uses_group_projection_fallback
    assert diagnostics.has_target_calibration


def test_cross_window_adapter_warns():
    diagnostics = make_alignment_diagnostics(
        method="hyperalignment",
        sample_mode="class_repetition",
        n_alignment_rows=30,
        n_components=16,
        actual_components=16,
        cross_window_adapter_used=True,
    )

    with pytest.warns(AlignmentDiagnosticWarning, match="cross-window projection adapter"):
        warn_for_alignment_diagnostics(diagnostics)


def test_requested_component_count_validates_positive_values():
    assert requested_component_count(64) == 64
    assert requested_component_count(float("inf")) > 1_000_000
    with pytest.raises(ValueError, match="positive"):
        requested_component_count(0)
    with pytest.raises(ValueError, match="integer component count"):
        requested_component_count(0.95)
