from __future__ import annotations

from pathlib import Path

import scipy.io as sio

from neureptrace.fieldtrip_mat import load_fieldtrip_raw_mat_epochs
from neureptrace.io.synthetic_fieldtrip import (
    SyntheticFieldTripConfig,
    make_synthetic_fieldtrip_data,
    write_synthetic_fieldtrip_dataset,
)


def _small_config(**overrides) -> SyntheticFieldTripConfig:
    values = {
        "participant_id": 7,
        "n_classes": 2,
        "main_repeats_per_class": 4,
        "cue_repeats_per_class": 2,
        "n_channels": 4,
        "n_times": 101,
        "tmax": 0.5,
        "noise_scale": 0.01,
        "alpha_scale": 0.0,
    }
    values.update(overrides)
    return SyntheticFieldTripConfig(**values)


def test_write_synthetic_fieldtrip_dataset_creates_mat_files(tmp_path: Path):
    output = write_synthetic_fieldtrip_dataset(tmp_path, _small_config())

    assert output.main_path.exists()
    assert output.cue_path is not None and output.cue_path.exists()
    assert output.manifest_path is not None and output.manifest_path.exists()
    assert output.main_trials == 8
    assert output.cue_trials == 4

    data = sio.loadmat(output.main_path)["data"]
    assert set(data.dtype.names) >= {"trial", "time", "trialinfo", "label", "grad", "sampleinfo"}

    epochs, metadata = load_fieldtrip_raw_mat_epochs(output.main_path)
    assert epochs.get_data(copy=False).shape == (8, 4, 101)
    assert metadata["condition"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert metadata["sample_start"].tolist() == [1, 102, 203, 304, 405, 506, 607, 708]


def test_write_synthetic_fieldtrip_dataset_refuses_to_overwrite_by_default(tmp_path: Path):
    config = _small_config(main_repeats_per_class=1, cue_repeats_per_class=1)
    write_synthetic_fieldtrip_dataset(tmp_path, config)

    try:
        write_synthetic_fieldtrip_dataset(tmp_path, config)
    except FileExistsError:
        pass
    else:  # pragma: no cover - defensive assertion style keeps failure message clear
        raise AssertionError("Expected FileExistsError when generated files already exist.")

    output = write_synthetic_fieldtrip_dataset(tmp_path, config, overwrite=True)
    assert output.main_path.exists()


def test_make_synthetic_fieldtrip_data_cli(tmp_path: Path):
    exit_code = make_synthetic_fieldtrip_data(
        [
            "--out",
            str(tmp_path),
            "--participant",
            "3",
            "--classes",
            "2",
            "--main-repeats",
            "3",
            "--cue-repeats",
            "2",
            "--channels",
            "4",
            "--times",
            "101",
            "--tmax",
            "0.5",
            "--seed",
            "11",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "Part3Data.mat").exists()
    assert (tmp_path / "Part3CueData.mat").exists()
    assert (tmp_path / "synthetic_fieldtrip_manifest.json").exists()
