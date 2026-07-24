from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.io.synthetic_fieldtrip import SyntheticFieldTripConfig, write_synthetic_fieldtrip_dataset


def _small_config(**overrides) -> SyntheticFieldTripConfig:
    values = {
        "participant_id": 7,
        "n_classes": 2,
        "main_repeats_per_class": 1,
        "cue_repeats_per_class": 1,
        "n_channels": 2,
        "n_times": 11,
        "noise_scale": 0.01,
        "alpha_scale": 0.0,
    }
    values.update(overrides)
    return SyntheticFieldTripConfig(**values)


@pytest.mark.parametrize(
    "config",
    [
        _small_config(
            main_file_template="Part{participant}.mat",
            cue_file_template="Part{participant}.mat",
            manifest_name=None,
        ),
        _small_config(
            cue_repeats_per_class=0,
            manifest_name="Part7Data.mat",
        ),
    ],
    ids=("main-cue", "main-manifest"),
)
def test_write_synthetic_fieldtrip_dataset_rejects_colliding_output_paths(
    tmp_path: Path,
    config: SyntheticFieldTripConfig,
):
    with pytest.raises(ValueError, match="output paths must be distinct"):
        write_synthetic_fieldtrip_dataset(tmp_path, config)

    assert not any(tmp_path.iterdir())
