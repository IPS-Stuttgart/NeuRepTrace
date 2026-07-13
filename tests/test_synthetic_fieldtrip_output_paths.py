from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.io.synthetic_fieldtrip import (
    SyntheticFieldTripConfig,
    write_synthetic_dataset,
    write_synthetic_fieldtrip_dataset,
)


def _small_config(**overrides: object) -> SyntheticFieldTripConfig:
    values: dict[str, object] = {
        "participant_id": 7,
        "n_classes": 2,
        "main_repeats_per_class": 1,
        "cue_repeats_per_class": 1,
        "n_channels": 2,
        "n_times": 11,
        "tmin": -0.1,
        "tmax": 0.3,
        "stimulus_window": (0.1, 0.2),
    }
    values.update(overrides)
    return SyntheticFieldTripConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "first_field", "second_field"),
    [
        (
            {
                "main_file_template": "shared.mat",
                "cue_file_template": "shared.mat",
            },
            "main_file_template",
            "cue_file_template",
        ),
        (
            {
                "main_file_template": "shared.mat",
                "manifest_name": "shared.mat",
            },
            "main_file_template",
            "manifest_name",
        ),
        (
            {
                "cue_file_template": "shared.mat",
                "manifest_name": "shared.mat",
            },
            "cue_file_template",
            "manifest_name",
        ),
    ],
)
def test_writer_rejects_colliding_output_paths(
    tmp_path: Path,
    overrides: dict[str, object],
    first_field: str,
    second_field: str,
):
    with pytest.raises(ValueError, match="output paths must be distinct") as exc_info:
        write_synthetic_fieldtrip_dataset(tmp_path, _small_config(**overrides))

    message = str(exc_info.value)
    assert first_field in message
    assert second_field in message
    assert list(tmp_path.iterdir()) == []


def test_compatibility_writer_alias_uses_output_collision_guard(tmp_path: Path):
    assert write_synthetic_dataset is write_synthetic_fieldtrip_dataset

    config = _small_config(
        main_file_template="nested/../shared.mat",
        manifest_name="shared.mat",
    )
    with pytest.raises(ValueError, match="output paths must be distinct"):
        write_synthetic_dataset(tmp_path, config)

    assert list(tmp_path.iterdir()) == []
