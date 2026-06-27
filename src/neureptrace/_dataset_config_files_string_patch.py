"""Normalize scalar FieldTrip dataset.files values."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_PATCH_MARKER = "_neureptrace_dataset_config_files_string_patch_installed"


def _with_scalar_files_normalized(config: Mapping[str, Any], dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    files = dataset.get("files")
    if not isinstance(files, (str, Path)):
        return config
    normalized = dict(config)
    normalized["dataset"] = {**dataset, "files": [files]}
    return normalized


def install() -> None:
    """Install scalar files normalization for FieldTrip configs."""

    dataset_config = importlib.import_module("neureptrace.dataset_config")
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
