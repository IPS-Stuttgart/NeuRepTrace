"""Normalize scalar FieldTrip dataset.files values and validate optional config sections."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

_PATCH_MARKER = "_neureptrace_dataset_config_files_string_patch_installed"
_SECTION_PATCH_MARKER = "_neureptrace_dataset_config_section_validation_patch_installed"


def _with_scalar_files_normalized(config: Mapping[str, Any], dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    files = dataset.get("files")
    if not isinstance(files, (str, Path)):
        return config
    normalized = dict(config)
    normalized["dataset"] = {**dataset, "files": [files]}
    return normalized


def _optional_section(config: Mapping[str, Any], name: str, *, error_type: type[Exception]) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise error_type(f"Config section '{name}' must be a mapping.")
    return dict(value)


def _install_optional_section_validation(dataset_config: Any) -> None:
    if getattr(dataset_config, _SECTION_PATCH_MARKER, False):
        return

    error_type = dataset_config.ConfigValidationError
    original_validate_dataset_config = dataset_config.validate_dataset_config

    def _metadata_section(config: Mapping[str, Any]) -> dict[str, Any]:
        return _optional_section(config, "metadata", error_type=error_type)

    def _validation_section(config: Mapping[str, Any]) -> dict[str, Any]:
        return _optional_section(config, "validation", error_type=error_type)

    def _participant_ids(config: Mapping[str, Any]) -> list[int | str]:
        participants = _optional_section(config, "participants", error_type=error_type)
        return dataset_config.parse_participant_ids(participants.get("ids"))

    @wraps(original_validate_dataset_config)
    def validate_dataset_config(
        config: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
        check_files: bool = False,
    ) -> list[str]:
        _optional_section(config, "participants", error_type=error_type)
        _optional_section(config, "decoding", error_type=error_type)
        _optional_section(config, "workflow", error_type=error_type)
        return original_validate_dataset_config(config, base_dir=base_dir, check_files=check_files)

    dataset_config._metadata_section = _metadata_section
    dataset_config._validation_section = _validation_section
    dataset_config._participant_ids = _participant_ids
    dataset_config.validate_dataset_config = validate_dataset_config
    setattr(dataset_config, _SECTION_PATCH_MARKER, True)


def install() -> None:
    """Install scalar files normalization and strict optional-section validation."""

    dataset_config = importlib.import_module("neureptrace.dataset_config")
    _install_optional_section_validation(dataset_config)
    if getattr(dataset_config, _PATCH_MARKER, False):
        return

    original_iter_dataset_files = dataset_config.iter_dataset_files
    original_fieldtrip_file_specs = dataset_config._fieldtrip_file_specs

    def iter_dataset_files(config: Mapping[str, Any], *, base_dir: str | Path = ".") -> list[Path]:
        dataset = dataset_config._dataset_section(config)
        if dataset.get("type") == "fieldtrip_mat":
            config = _with_scalar_files_normalized(config, dataset)
        return original_iter_dataset_files(config, base_dir=base_dir)

    def _fieldtrip_file_specs(config: Mapping[str, Any], *, base_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
        dataset = dataset_config._dataset_section(config)
        config = _with_scalar_files_normalized(config, dataset)
        return original_fieldtrip_file_specs(config, base_dir=base_dir)

    dataset_config.iter_dataset_files = iter_dataset_files
    dataset_config._fieldtrip_file_specs = _fieldtrip_file_specs
    setattr(dataset_config, _PATCH_MARKER, True)


__all__ = ["install"]
