import neureptrace  # noqa: F401  # install runtime alignment patches

from neureptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SELECTION
from neureptrace.decoding.source_alignment import (
    PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
    TARGET_CALIBRATED_ALIGNMENT,
    _source_alignment_repetition_selection,
    source_alignment_config,
)


def test_pseudo_label_target_calibrated_class_repetition_uses_first_source_offsets():
    config = source_alignment_config(
        method="mcca",
        anchor_mode="class_repetition",
        target_projection=PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
        target_calibration_per_anchor=2,
    )

    assert _source_alignment_repetition_selection(config, "class_repetition") == "first"


def test_target_calibrated_class_repetition_still_uses_first_source_offsets():
    config = source_alignment_config(
        method="mcca",
        anchor_mode="class_repetition",
        target_projection=TARGET_CALIBRATED_ALIGNMENT,
        target_calibration_per_anchor=2,
    )

    assert _source_alignment_repetition_selection(config, "class_repetition") == "first"


def test_strict_source_class_repetition_keeps_default_random_offsets():
    config = source_alignment_config(
        method="mcca",
        anchor_mode="class_repetition",
    )

    assert _source_alignment_repetition_selection(config, "class_repetition") == DEFAULT_CLASS_LIMIT_SELECTION
