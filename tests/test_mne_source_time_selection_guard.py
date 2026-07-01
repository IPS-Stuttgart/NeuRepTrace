from pathlib import Path

import pytest

from neureptrace.mne_time_decode import run_time_resolved_decode


def _run_missing_epochs_with_source_time_selection(**kwargs):
    return run_time_resolved_decode(
        Path("missing-epo.fif"),
        "condition",
        Path("out.csv"),
        source_time_selection="source_oof_best_time",
        **kwargs,
    )


def test_source_time_selection_rejects_alignment_before_loading_epochs():
    with pytest.raises(ValueError, match="source_time_selection.*source alignment"):
        _run_missing_epochs_with_source_time_selection(alignment_method="procrustes")


def test_source_time_selection_rejects_pseudo_label_self_training_before_loading_epochs():
    with pytest.raises(ValueError, match="source_time_selection.*pseudo_label_self_training"):
        _run_missing_epochs_with_source_time_selection(pseudo_label_self_training=True)


def test_source_time_selection_rejects_dann_before_loading_epochs():
    with pytest.raises(ValueError, match="source_time_selection.*DANN"):
        _run_missing_epochs_with_source_time_selection(decoder="domain-adversarial-neural-network")
