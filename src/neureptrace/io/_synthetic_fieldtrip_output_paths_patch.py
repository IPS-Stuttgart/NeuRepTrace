"""Reject colliding synthetic FieldTrip output paths."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

_MARKER = "_neureptrace_synthetic_fieldtrip_output_paths_installed"
_TARGET = "_neureptrace_synthetic_fieldtrip_output_paths_target"


def _configured_output_paths(
    data_dir: str | Path,
    config: Any,
    *,
    write_manifest: bool,
) -> list[tuple[str, Path]]:
    output_dir = Path(data_dir)
    participant_id = str(config.participant_id)
    paths = [
        (
            "main_file_template",
            output_dir / config.main_file_template.format(participant=participant_id),
        )
    ]
    if config.cue_file_template is not None and config.cue_repeats_per_class > 0:
        paths.append(
            (
                "cue_file_template",
                output_dir / config.cue_file_template.format(participant=participant_id),
            )
        )
    if write_manifest and config.manifest_name:
        paths.append(("manifest_name", output_dir / config.manifest_name))
    return paths


def _validate_distinct_output_paths(
    data_dir: str | Path,
    config: Any,
    *,
    write_manifest: bool,
) -> None:
    seen: dict[Path, str] = {}
    for field_name, path in _configured_output_paths(
        data_dir,
        config,
        write_manifest=write_manifest,
    ):
        resolved = path.resolve(strict=False)
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                "Synthetic FieldTrip output paths must be distinct; "
                f"{previous} and {field_name} both resolve to {resolved}."
            )
        seen[resolved] = field_name


def install() -> None:
    import neureptrace.io.synthetic_fieldtrip as synthetic_fieldtrip

    current = synthetic_fieldtrip.write_synthetic_fieldtrip_dataset
    if getattr(current, _MARKER, False):
        synthetic_fieldtrip.write_synthetic_dataset = current
        return

    original = current

    @wraps(original)
    def write_synthetic_fieldtrip_dataset(
        data_dir: str | Path,
        config: Any = None,
        *,
        overwrite: bool = False,
        write_manifest: bool = True,
    ) -> Any:
        effective_config = config or synthetic_fieldtrip.SyntheticFieldTripConfig()
        synthetic_fieldtrip._validate_config(effective_config)  # pylint: disable=protected-access
        _validate_distinct_output_paths(
            data_dir,
            effective_config,
            write_manifest=write_manifest,
        )

        current_target = synthetic_fieldtrip.write_synthetic_fieldtrip_dataset
        target = getattr(write_synthetic_fieldtrip_dataset, _TARGET)
        if current_target is not write_synthetic_fieldtrip_dataset:
            target = current_target
        return target(
            data_dir,
            effective_config,
            overwrite=overwrite,
            write_manifest=write_manifest,
        )

    setattr(write_synthetic_fieldtrip_dataset, _MARKER, True)
    setattr(write_synthetic_fieldtrip_dataset, _TARGET, original)
    synthetic_fieldtrip.write_synthetic_fieldtrip_dataset = write_synthetic_fieldtrip_dataset
    synthetic_fieldtrip.write_synthetic_dataset = write_synthetic_fieldtrip_dataset


__all__ = ["install"]
