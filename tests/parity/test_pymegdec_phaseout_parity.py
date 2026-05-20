from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.parity


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_file(path: Path | None, *, env_names: str) -> Path:
    if path is None:
        pytest.skip(f"Set {env_names} to enable this PyMEGDec phase-out parity test.")
    if not path.is_file():
        pytest.skip(f"Parity input file does not exist: {path}")
    return path


def _fieldtrip_loader():
    module = pytest.importorskip(
        "neureptrace.fieldtrip_mat",
        reason="FieldTrip MAT parity tests require the NeuRepTrace FieldTrip loader from the PyMEGDec phase-out migration.",
    )
    for name in (
        "load_fieldtrip_raw_mat_epochs",
        "read_fieldtrip_raw_mat_epochs",
        "read_fieldtrip_raw_mat_as_epochs",
    ):
        loader = getattr(module, name, None)
        if loader is not None:
            return loader
    pytest.skip("neureptrace.fieldtrip_mat does not expose a supported FieldTrip MAT epochs loader.")


def _coerce_epochs_and_metadata(result: Any) -> tuple[mne.BaseEpochs, pd.DataFrame]:
    if isinstance(result, tuple) and len(result) == 2:
        epochs, metadata = result
    else:
        epochs = result
        metadata = getattr(epochs, "metadata", None)
    if metadata is None:
        raise AssertionError("FieldTrip MAT loader must provide metadata, either returned explicitly or attached to epochs.metadata.")
    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.DataFrame(metadata)
    return epochs, metadata.reset_index(drop=True)


def _load_fieldtrip(path: Path, **kwargs: Any) -> tuple[mne.BaseEpochs, pd.DataFrame, list[warnings.WarningMessage]]:
    loader = _fieldtrip_loader()
    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        result = loader(path, **kwargs)
    epochs, metadata = _coerce_epochs_and_metadata(result)
    return epochs, metadata, list(warning_record)


def _condition_column(metadata: pd.DataFrame) -> str:
    for column in ("condition", "label", "trialinfo", "trialinfo_0"):
        if column in metadata.columns:
            return column
    raise AssertionError(f"No condition-like label column found in metadata columns: {list(metadata.columns)}")


def _data_array(epochs: mne.BaseEpochs) -> np.ndarray:
    try:
        return epochs.get_data(copy=False)
    except TypeError:
        return epochs.get_data()


def _assert_basic_epochs_invariants(epochs: mne.BaseEpochs, metadata: pd.DataFrame) -> None:
    data = _data_array(epochs)
    assert data.ndim == 3
    assert data.shape[0] == len(epochs)
    assert data.shape[0] == len(metadata)
    assert data.shape[1] == len(epochs.ch_names)
    assert data.shape[2] == len(epochs.times)
    assert np.all(np.isfinite(data))
    assert np.all(np.diff(epochs.times) > 0)
    assert len(set(epochs.ch_names)) == len(epochs.ch_names)

    condition = metadata[_condition_column(metadata)]
    assert condition.notna().all()
    assert condition.nunique() >= 2

    if "sample_start" in metadata.columns and "sample_stop" in metadata.columns:
        assert (metadata["sample_stop"].to_numpy() >= metadata["sample_start"].to_numpy()).all()
        assert np.all(np.diff(metadata["sample_start"].to_numpy()) >= 0)


def _assert_expected_shape(epochs: mne.BaseEpochs) -> None:
    expected_trials = _env_int("NEUREPTRACE_PARITY_EXPECT_N_TRIALS")
    expected_channels = _env_int("NEUREPTRACE_PARITY_EXPECT_N_CHANNELS")
    expected_times = _env_int("NEUREPTRACE_PARITY_EXPECT_N_TIMES")
    data = _data_array(epochs)
    if expected_trials is not None:
        assert data.shape[0] == expected_trials
    if expected_channels is not None:
        assert data.shape[1] == expected_channels
    if expected_times is not None:
        assert data.shape[2] == expected_times


def _assert_expected_class_balance(metadata: pd.DataFrame) -> None:
    expected_classes = _env_int("NEUREPTRACE_PARITY_EXPECT_N_CLASSES")
    expected_trials_per_class = _env_int("NEUREPTRACE_PARITY_EXPECT_TRIALS_PER_CLASS")
    condition = metadata[_condition_column(metadata)]
    counts = condition.value_counts(dropna=False)
    if expected_classes is not None:
        assert len(counts) == expected_classes
    if expected_trials_per_class is not None:
        assert set(counts.astype(int).tolist()) == {expected_trials_per_class}


def _assert_expected_trim_warnings(warning_record: list[warnings.WarningMessage]) -> None:
    if not _env_bool("NEUREPTRACE_PARITY_EXPECT_TRIM_WARNING"):
        return
    warning_text = "\n".join(str(item.message) for item in warning_record)
    assert "Trimming FieldTrip" in warning_text
    assert "label" in warning_text


def _metadata_values(metadata: pd.DataFrame) -> Mapping[str, np.ndarray]:
    condition_column = _condition_column(metadata)
    values: dict[str, np.ndarray] = {"condition": metadata[condition_column].to_numpy()}
    for column in ("sample_start", "sample_stop", "trial", "trial_index"):
        if column in metadata.columns:
            values[column] = metadata[column].to_numpy()
    return values


def test_fieldtrip_mat_loader_preserves_bush_trial_geometry_and_labels() -> None:
    """Acceptance parity check for a PyMEGDec/Bush FieldTrip raw MAT file.

    This is opt-in because the real ``Part*Data.mat`` files are private. It is
    intended to be run before retiring PyMEGDec, for example:

    ``NEUREPTRACE_PARITY_MAIN_MAT=D:/Uni-Data/Bush_MEG-Data/MEG-Data/Part10Data.mat``
    """

    mat_path = _require_file(
        _env_path("NEUREPTRACE_PARITY_MAIN_MAT", "PYMEGDEC_PARITY_MAIN_MAT"),
        env_names="NEUREPTRACE_PARITY_MAIN_MAT or PYMEGDEC_PARITY_MAIN_MAT",
    )

    epochs, metadata, warning_record = _load_fieldtrip(mat_path)

    _assert_basic_epochs_invariants(epochs, metadata)
    _assert_expected_shape(epochs)
    _assert_expected_class_balance(metadata)
    _assert_expected_trim_warnings(warning_record)


def test_fieldtrip_main_and_cue_files_are_transfer_compatible() -> None:
    """Check that migrated main/cue inputs can feed a NeuRepTrace transfer workflow."""

    main_path = _require_file(
        _env_path("NEUREPTRACE_PARITY_MAIN_MAT", "PYMEGDEC_PARITY_MAIN_MAT"),
        env_names="NEUREPTRACE_PARITY_MAIN_MAT or PYMEGDEC_PARITY_MAIN_MAT",
    )
    cue_path = _require_file(
        _env_path("NEUREPTRACE_PARITY_CUE_MAT", "PYMEGDEC_PARITY_CUE_MAT"),
        env_names="NEUREPTRACE_PARITY_CUE_MAT or PYMEGDEC_PARITY_CUE_MAT",
    )

    main_epochs, main_metadata, _ = _load_fieldtrip(main_path)
    cue_epochs, cue_metadata, _ = _load_fieldtrip(cue_path)

    _assert_basic_epochs_invariants(main_epochs, main_metadata)
    _assert_basic_epochs_invariants(cue_epochs, cue_metadata)

    assert main_epochs.ch_names == cue_epochs.ch_names
    assert np.allclose(main_epochs.times, cue_epochs.times)
    assert _data_array(main_epochs).shape[1:] == _data_array(cue_epochs).shape[1:]
    assert set(main_metadata[_condition_column(main_metadata)]) == set(cue_metadata[_condition_column(cue_metadata)])


def test_fieldtrip_mat_loader_matches_reference_mne_epochs_artifact() -> None:
    """Compare migrated FieldTrip loading against a pre-staged reference FIF artifact.

    Create the reference once with the legacy PyMEGDec/MNE staging path, then run
    this test with ``NEUREPTRACE_PARITY_REFERENCE_EPOCHS=/path/to/reference-epo.fif``.
    """

    mat_path = _require_file(
        _env_path("NEUREPTRACE_PARITY_MAIN_MAT", "PYMEGDEC_PARITY_MAIN_MAT"),
        env_names="NEUREPTRACE_PARITY_MAIN_MAT or PYMEGDEC_PARITY_MAIN_MAT",
    )
    reference_path = _require_file(
        _env_path("NEUREPTRACE_PARITY_REFERENCE_EPOCHS", "PYMEGDEC_PARITY_REFERENCE_EPOCHS"),
        env_names="NEUREPTRACE_PARITY_REFERENCE_EPOCHS or PYMEGDEC_PARITY_REFERENCE_EPOCHS",
    )

    migrated_epochs, migrated_metadata, _ = _load_fieldtrip(mat_path)
    reference_epochs = mne.read_epochs(reference_path, preload=True, verbose="error")
    reference_metadata = reference_epochs.metadata
    if reference_metadata is None:
        pytest.skip("Reference MNE epochs artifact has no metadata to compare.")
    reference_metadata = reference_metadata.reset_index(drop=True)

    migrated_data = _data_array(migrated_epochs)
    reference_data = _data_array(reference_epochs)
    assert migrated_data.shape == reference_data.shape
    assert migrated_epochs.ch_names == reference_epochs.ch_names
    assert np.allclose(migrated_epochs.times, reference_epochs.times)
    assert np.allclose(migrated_data, reference_data, rtol=1e-7, atol=1e-10)

    migrated_values = _metadata_values(migrated_metadata)
    reference_values = _metadata_values(reference_metadata)
    assert migrated_values.keys() <= reference_values.keys() or reference_values.keys() <= migrated_values.keys()
    for key in sorted(set(migrated_values) & set(reference_values)):
        assert np.array_equal(migrated_values[key], reference_values[key]), key
