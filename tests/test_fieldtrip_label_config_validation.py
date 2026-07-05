from __future__ import annotations

import argparse

import numpy as np
import pytest

import neureptrace.fieldtrip_mat as fieldtrip_mat


def _trialinfo_metadata(*, label_base=1, trialinfo_column=0):
    return fieldtrip_mat._metadata_from_trialinfo(
        n_trials=2,
        trialinfo=np.asarray([[1, 10], [2, 20]]),
        sampleinfo=None,
        label_column="condition",
        label_base=label_base,
        trialinfo_column=trialinfo_column,
    )


def test_fieldtrip_parse_label_base_rejects_boolean_values() -> None:
    for value in (True, False, np.bool_(True), np.array(True)):
        with pytest.raises(argparse.ArgumentTypeError, match="label-base"):
            fieldtrip_mat._parse_label_base(value)


def test_fieldtrip_metadata_rejects_boolean_label_base() -> None:
    for value in (True, False, np.bool_(True), np.array(False)):
        with pytest.raises(ValueError, match="label_base.*boolean"):
            _trialinfo_metadata(label_base=value)


def test_fieldtrip_metadata_rejects_boolean_trialinfo_column() -> None:
    for value in (True, False, np.bool_(True), np.array(False)):
        with pytest.raises(ValueError, match="trialinfo_column.*boolean"):
            _trialinfo_metadata(trialinfo_column=value)


def test_fieldtrip_metadata_accepts_numeric_numpy_scalars() -> None:
    metadata = _trialinfo_metadata(label_base=np.array(1.0), trialinfo_column=np.array(0))

    assert metadata["condition"].tolist() == [0, 1]
    assert metadata["trialinfo_1"].tolist() == [10, 20]
