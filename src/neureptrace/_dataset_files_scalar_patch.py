"""Handle scalar FieldTrip ``dataset.files`` entries as one file spec.

The config loader accepts ``dataset.files`` as the direct-file alternative to
participant templates.  Before this patch, the FieldTrip paths branch iterated
``dataset.files`` directly.  A scalar string was therefore split into path
characters, and a single mapping such as ``{path: file.mat, split: train}`` was
split into mapping keys.  This patch normalizes scalar file specs before both
path rendering and loader-spec construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_PATCH_MARKER = "_neureptrace_dataset_files_scalar_patch_installed"


def _file_entries(files: Any, *, config_error: type[Exception]) -> list[Any]:
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        return [files] if str(files).strip() else []
    if isinstance(files, Mapping):
        return [files]
    if isinstance(files, (bytes, bytearray)):
        raise config_error("dataset.files must be a path, mapping, or sequence of paths/mappings.")
    if isinstance(files, Iterable):
        return list(files)
    raise config_error("dataset.files must be a path, mapping, or sequence of paths/mappings.")


def _entry_path_and_extra(entry: Any, *, config_error: type[Exception]) -> tuple[Any, dict[str, Any]]:
    if isinstance(entry, Mapping):
        value = entry.get("path") or entry.get("file")
        extra = {str(key): item for key, item in entry.items() if key not in {"path", "file"}}
    else:
        value = entry
        extra = {}
    if value is None or str(value).strip() == "":
        raise config_error("dataset.files entries must contain a path.")
    return value, extra


def install() -> None:
    """Install scalar ``dataset.files`` normalization for FieldTrip configs."""

    from neureptrace import dataset_config

    if getattr(dataset_config, _PATCH_MARKER, False):
        return

    original_validate_dataset_config = dataset_config.validate_dataset_config
    original_iter_dataset_files = dataset_config.iter_dataset_files
    original_fieldtrip_file_specs = dataset_config._fieldtrip_file_specs
    config_error = dataset_config.ConfigValidationError

    def validate_dataset_config(config: Mapping[str, Any], *, base_dir: str | Path = ".", check_files: bool = False) -> list[str]:
        dataset = dataset_config._dataset_section(config)
        if dataset.get("type") == "fieldtrip_mat" and "files" in dataset:
            for entry in _file_entries(dataset.get("files"), config_error=config_error):
                _entry_path_and_extra(entry, config_error=config_error)
        return original_validate_dataset_config(config, base_dir=base_dir, check_files=check_files)

    def iter_dataset_files(config: Mapping[str, Any], *, base_dir: str | Path = ".") -> list[Path]:
        dataset = dataset_config._dataset_section(config)
        if dataset.get("type") != "fieldtrip_mat":
            return original_iter_dataset_files(config, base_dir=base_dir)
        files = _file_entries(dataset.get("files"), config_error=config_error)
        if not files:
            return original_iter_dataset_files(config, base_dir=base_dir)
        root = dataset.get("root")
        resolved: list[Path] = []
        for entry in files:
            value, _extra = _entry_path_and_extra(entry, config_error=config_error)
            resolved.append(dataset_config.expand_path(value, base_dir=base_dir, root=root))
        return resolved

    def _fieldtrip_file_specs(config: Mapping[str, Any], *, base_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
        dataset = dataset_config._dataset_section(config)
        files = _file_entries(dataset.get("files"), config_error=config_error)
        if not files:
            return original_fieldtrip_file_specs(config, base_dir=base_dir)
        root = dataset.get("root")
        specs: list[tuple[Path, dict[str, Any]]] = []
        for entry in files:
            value, extra = _entry_path_and_extra(entry, config_error=config_error)
            specs.append((dataset_config.expand_path(value, base_dir=base_dir, root=root), extra))
        return specs

    validate_dataset_config.__doc__ = original_validate_dataset_config.__doc__
    iter_dataset_files.__doc__ = original_iter_dataset_files.__doc__
    _fieldtrip_file_specs.__doc__ = original_fieldtrip_file_specs.__doc__
    dataset_config.validate_dataset_config = validate_dataset_config
    dataset_config.iter_dataset_files = iter_dataset_files
    dataset_config._fieldtrip_file_specs = _fieldtrip_file_specs
    setattr(dataset_config, _PATCH_MARKER, True)
