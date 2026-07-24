"""Prevent synthetic FieldTrip outputs from overwriting one another.

The synthetic-data writer supports configurable main, cue, and manifest file
names. Distinct configuration fields can resolve to the same filesystem path,
including through ``..`` components or platform-specific case folding. Without
an explicit guard, the later cue or manifest write silently replaces an earlier
output created by the same call.
"""

from __future__ import annotations

import importlib
import os
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any

_PATCH_MARKER = "_neureptrace_synthetic_fieldtrip_output_collision_patch_installed"


def _active_output_paths(synthetic_fieldtrip: Any, data_dir: str | Path, config: Any, *, write_manifest: bool) -> dict[str, Path]:
    output_dir = Path(data_dir)
    participant_id = str(config.participant_id)
    paths = {
        "main_path": output_dir / synthetic_fieldtrip._format_participant_file(config.main_file_template, participant_id),
    }
    if config.cue_file_template is not None and config.cue_repeats_per_class > 0:
        paths["cue_path"] = output_dir / synthetic_fieldtrip._format_participant_file(config.cue_file_template, participant_id)
    if write_manifest and config.manifest_name:
        paths["manifest_path"] = output_dir / config.manifest_name
    return paths


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _validate_distinct_output_paths(synthetic_fieldtrip: Any, data_dir: str | Path, config: Any, *, write_manifest: bool) -> None:
    paths = _active_output_paths(synthetic_fieldtrip, data_dir, config, write_manifest=write_manifest)
    by_destination: dict[str, list[str]] = defaultdict(list)
    for label, path in paths.items():
        by_destination[_normalized_path(path)].append(label)

    collisions = {destination: labels for destination, labels in by_destination.items() if len(labels) > 1}
    if not collisions:
        return

    details = "; ".join(f"{', '.join(labels)} -> {destination}" for destination, labels in sorted(collisions.items()))
    raise ValueError(f"Synthetic FieldTrip output paths must be distinct; conflicting outputs: {details}.")


def install() -> None:
    """Install a pre-write collision check on the synthetic dataset writer."""

    synthetic_fieldtrip = importlib.import_module("neureptrace.io.synthetic_fieldtrip")
    if getattr(synthetic_fieldtrip, _PATCH_MARKER, False):
        return

    original_write = synthetic_fieldtrip.write_synthetic_fieldtrip_dataset

    @wraps(original_write)
    def write_synthetic_fieldtrip_dataset(
        data_dir: str | Path,
        config: Any = None,
        *,
        overwrite: bool = False,
        write_manifest: bool = True,
    ):
        effective_config = config or synthetic_fieldtrip.SyntheticFieldTripConfig()
        synthetic_fieldtrip._validate_config(effective_config)
        _validate_distinct_output_paths(
            synthetic_fieldtrip,
            data_dir,
            effective_config,
            write_manifest=write_manifest,
        )
        return original_write(
            data_dir,
            effective_config,
            overwrite=overwrite,
            write_manifest=write_manifest,
        )

    synthetic_fieldtrip.write_synthetic_fieldtrip_dataset = write_synthetic_fieldtrip_dataset
    synthetic_fieldtrip.write_synthetic_dataset = write_synthetic_fieldtrip_dataset
    setattr(synthetic_fieldtrip, _PATCH_MARKER, True)


__all__ = ["install"]
