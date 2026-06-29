from __future__ import annotations

import numpy as np
import pytest

from neureptrace import mne_time_decode, mne_time_decode_ensemble, mne_time_decode_foldlocal, time_transfer_decode
from neureptrace._mne_time_decode_float_sequence_validation_patch import _normalize_bool, _normalize_temporal_train_window_value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (True, True),
        (False, False),
        (np.bool_(False), False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        ("true", True),
        ("false", False),
        ("YES", True),
        ("off", False),
        (np.asarray(True), True),
    ],
)
def test_time_decode_boolean_controls_normalize_common_tokens(value, expected) -> None:
    assert _normalize_bool(value, name="label_shuffle_control") is expected


@pytest.mark.parametrize("value", ["maybe", "", 2, -1, 0.5, np.inf, np.asarray([False, True])])
def test_time_decode_boolean_controls_reject_ambiguous_tokens(value) -> None:
    with pytest.raises(ValueError, match="label_shuffle_control"):
        _normalize_bool(value, name="label_shuffle_control")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ((0.0, 0.1), (0.0, 0.1)),
        ([0, 0.1], (0.0, 0.1)),
        (np.asarray([0.0, 0.1]), (0.0, 0.1)),
        ("0 0.1", (0.0, 0.1)),
        ("0, 0.1", (0.0, 0.1)),
    ],
)
def test_temporal_train_window_normalizes_common_tokens(value, expected) -> None:
    assert _normalize_temporal_train_window_value(value) == expected
    assert mne_time_decode._normalize_temporal_train_window(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        (True, 0.1),
        (0.0, False),
        np.asarray([0.0, np.inf]),
        (0.0, np.nan),
        (0.2, 0.1),
        (0.0,),
        "0 0.1 0.2",
    ],
)
def test_temporal_train_window_rejects_malformed_controls(value) -> None:
    with pytest.raises(ValueError, match="temporal_train_window"):
        _normalize_temporal_train_window_value(value)
    with pytest.raises(ValueError, match="temporal_train_window"):
        mne_time_decode._normalize_temporal_train_window(value)


def test_base_time_decode_rejects_ambiguous_label_shuffle_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="label_shuffle_control"):
        mne_time_decode.run_time_resolved_decode(
            tmp_path / "missing-epo.fif",
            "condition",
            tmp_path / "out.csv",
            label_shuffle_control="maybe",
        )


def test_base_time_decode_rejects_ambiguous_pseudo_label_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="pseudo_label_self_training"):
        mne_time_decode.run_time_resolved_decode(
            tmp_path / "missing-epo.fif",
            "condition",
            tmp_path / "out.csv",
            pseudo_label_self_training="maybe",
        )


def test_base_time_decode_rejects_bad_temporal_train_window_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="temporal_train_window"):
        mne_time_decode.run_time_resolved_decode(
            tmp_path / "missing-epo.fif",
            "condition",
            tmp_path / "out.csv",
            temporal_train_window=(True, 0.1),
        )


def test_time_transfer_decode_rejects_bad_temporal_train_window_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="temporal_train_window"):
        time_transfer_decode.run_time_transfer_decode(
            train_epochs_path=tmp_path / "missing-train-epo.fif",
            validation_epochs_path=tmp_path / "missing-validation-epo.fif",
            label_column="condition",
            out_path=tmp_path / "transfer.csv",
            temporal_train_window=(True, 0.1),
        )


def test_foldlocal_time_decode_rejects_ambiguous_label_shuffle_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="label_shuffle_control"):
        mne_time_decode_foldlocal.run_time_resolved_decode(
            tmp_path / "missing-epo.fif",
            "condition",
            tmp_path / "out.csv",
            label_shuffle_control="maybe",
        )


def test_ensemble_time_decode_rejects_ambiguous_pseudo_label_before_mode_checks(tmp_path) -> None:
    with pytest.raises(ValueError, match="pseudo_label_self_training"):
        mne_time_decode_ensemble.run_time_resolved_decode(
            tmp_path / "missing-epo.fif",
            "condition",
            tmp_path / "out.csv",
            decoder="logistic-svm-ensemble",
            pseudo_label_self_training="maybe",
        )


def test_ensemble_time_decode_rejects_ambiguous_source_debiasing_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="ensemble_source_baseline_debiasing"):
        mne_time_decode_ensemble.run_time_resolved_decode(
            tmp_path / "missing-epo.fif",
            "condition",
            tmp_path / "out.csv",
            decoder="logistic-svm-ensemble",
            ensemble_source_baseline_debiasing="maybe",
        )
