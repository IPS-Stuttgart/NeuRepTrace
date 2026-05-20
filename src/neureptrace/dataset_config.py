"""Config loading and validation for dataset-driven NeuRepTrace workflows."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.io.dataset import EpochDataset
from neureptrace.io.fieldtrip_mat import load_fieldtrip_mat_epochs

SUPPORTED_DATASET_TYPES = {"fieldtrip_mat", "mne_epochs"}
DEFAULT_SCHEMA_VERSION = "neureptrace.dataset.v1"


class ConfigValidationError(ValueError):
    """Raised when a dataset config is malformed."""


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("YAML configs require PyYAML. Install neureptrace with the yaml dependency enabled.") from exc
    return yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file."""

    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        loaded = _yaml_module().safe_load(text)
    else:
        raise ValueError(f"Unsupported config extension '{path.suffix}'. Use .json, .yml, or .yaml.")
    if not isinstance(loaded, dict):
        raise ValueError("Dataset config must load to a JSON/YAML object.")
    return loaded


def parse_scalar(value: str) -> Any:
    """Parse a CLI override value as JSON/YAML-ish scalar when possible."""

    stripped = value.strip()
    if stripped == "":
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    lowered = stripped.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def apply_overrides(config: Mapping[str, Any], overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Return a deep-copied config with ``key.path=value`` CLI overrides applied."""

    updated = copy.deepcopy(dict(config))
    for override in overrides or ():
        if "=" not in override:
            raise ValueError(f"Override '{override}' must have the form dotted.path=value.")
        dotted_path, raw_value = override.split("=", 1)
        keys = [key for key in dotted_path.split(".") if key]
        if not keys:
            raise ValueError(f"Override '{override}' has an empty key path.")
        cursor: dict[str, Any] = updated
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot set override '{override}' because '{key}' is not a mapping.")
            cursor = child
        cursor[keys[-1]] = parse_scalar(raw_value)
    return updated


def parse_participant_ids(value: Any) -> list[int | str]:
    """Parse compact participant specifications such as ``1-4,6,8``."""

    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        parts: Iterable[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        parts = value
    else:
        raise ValueError("participants.ids must be an int, string, or list.")

    parsed: list[int | str] = []
    for part in parts:
        if isinstance(part, int):
            parsed.append(part)
            continue
        text = str(part).strip()
        if not text:
            continue
        if "," in text:
            parsed.extend(parse_participant_ids(text))
            continue
        if "-" in text and text.replace("-", "").isdigit():
            start_text, stop_text = text.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            step = 1 if stop >= start else -1
            parsed.extend(range(start, stop + step, step))
            continue
        try:
            parsed.append(int(text))
        except ValueError:
            parsed.append(text)
    return parsed


def expand_path(value: str | Path, *, base_dir: str | Path, root: str | Path | None = None) -> Path:
    """Expand env vars and resolve paths relative to root or config directory."""

    raw = os.path.expanduser(os.path.expandvars(str(value)))
    path = Path(raw)
    if path.is_absolute():
        return path
    if root is not None:
        root_path = Path(os.path.expanduser(os.path.expandvars(str(root))))
        if not root_path.is_absolute():
            root_path = Path(base_dir) / root_path
        return root_path / path
    return Path(base_dir) / path


def _dataset_section(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ConfigValidationError("Config must contain a 'dataset' mapping.")
    return dataset


def validate_dataset_config(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    check_files: bool = False,
) -> list[str]:
    """Validate a dataset config and return non-fatal warnings."""

    warnings: list[str] = []
    schema_version = config.get("schema_version", DEFAULT_SCHEMA_VERSION)
    if schema_version != DEFAULT_SCHEMA_VERSION:
        warnings.append(f"Unexpected schema_version '{schema_version}'; expected '{DEFAULT_SCHEMA_VERSION}'.")

    dataset = _dataset_section(config)
    dataset_type = str(dataset.get("type", "")).strip()
    if dataset_type not in SUPPORTED_DATASET_TYPES:
        raise ConfigValidationError(
            f"dataset.type must be one of {sorted(SUPPORTED_DATASET_TYPES)}; got {dataset_type!r}."
        )

    if dataset_type == "mne_epochs":
        if not (dataset.get("epochs") or dataset.get("epochs_file")):
            raise ConfigValidationError("mne_epochs configs require dataset.epochs or dataset.epochs_file.")
    if dataset_type == "fieldtrip_mat":
        participants = config.get("participants", {}) or {}
        has_participant_template = bool(dataset.get("participant_file") or dataset.get("file_template"))
        has_files = bool(dataset.get("files"))
        if not has_files and not has_participant_template:
            raise ConfigValidationError("fieldtrip_mat configs require dataset.files or dataset.participant_file.")
        if has_participant_template and not parse_participant_ids(participants.get("ids")):
            raise ConfigValidationError("fieldtrip_mat participant templates require participants.ids.")

    decoding = config.get("decoding", {}) or config.get("workflow", {}) or {}
    if isinstance(decoding, dict) and "label_column" not in decoding:
        warnings.append("No decoding.label_column was configured; decode-from-config will require one.")

    if check_files:
        for path in iter_dataset_files(config, base_dir=base_dir):
            if not path.exists():
                raise ConfigValidationError(f"Configured input file does not exist: {path}")
    return warnings


def iter_dataset_files(config: Mapping[str, Any], *, base_dir: str | Path = ".") -> list[Path]:
    """Return the input files referenced by a dataset config."""

    dataset = _dataset_section(config)
    dataset_type = dataset.get("type")
    root = dataset.get("root")
    if dataset_type == "mne_epochs":
        return [expand_path(dataset.get("epochs") or dataset.get("epochs_file"), base_dir=base_dir, root=root)]

    if dataset_type != "fieldtrip_mat":
        return []

    files = dataset.get("files")
    if files:
        resolved: list[Path] = []
        for item in files:
            if isinstance(item, dict):
                value = item.get("path") or item.get("file")
            else:
                value = item
            if value is None:
                raise ConfigValidationError("dataset.files entries must contain a path.")
            resolved.append(expand_path(value, base_dir=base_dir, root=root))
        return resolved

    participants = parse_participant_ids((config.get("participants", {}) or {}).get("ids"))
    template = dataset.get("participant_file") or dataset.get("file_template")
    return [expand_path(str(template).format(participant=participant), base_dir=base_dir, root=root) for participant in participants]


def _load_mne_epochs_dataset(config: Mapping[str, Any], *, base_dir: str | Path) -> EpochDataset:
    import mne

    dataset = _dataset_section(config)
    root = dataset.get("root")
    epochs_path = expand_path(dataset.get("epochs") or dataset.get("epochs_file"), base_dir=base_dir, root=root)
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="error")
    metadata = epochs.metadata.copy() if epochs.metadata is not None else None
    metadata_csv = dataset.get("metadata_csv")
    if metadata_csv is not None:
        metadata = pd.read_csv(expand_path(metadata_csv, base_dir=base_dir, root=root))
    if metadata is None:
        metadata = pd.DataFrame(index=range(len(epochs)))
    if len(metadata) != len(epochs):
        raise ValueError(f"Metadata row count ({len(metadata)}) does not match epochs ({len(epochs)}).")
    return EpochDataset(
        data=epochs.get_data(copy=True),
        times=epochs.times.copy(),
        channel_names=list(epochs.ch_names),
        metadata=metadata.reset_index(drop=True),
        name=str(dataset.get("name") or epochs_path.stem),
        provenance={"path": str(epochs_path), "loader": "mne_epochs"},
    )


def _fieldtrip_file_specs(config: Mapping[str, Any], *, base_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    dataset = _dataset_section(config)
    root = dataset.get("root")
    files = dataset.get("files")
    if files:
        specs: list[tuple[Path, dict[str, Any]]] = []
        for item in files:
            if isinstance(item, dict):
                value = item.get("path") or item.get("file")
                extra = {key: item[key] for key in item if key not in {"path", "file"}}
            else:
                value = item
                extra = {}
            if value is None:
                raise ConfigValidationError("dataset.files entries must contain a path.")
            specs.append((expand_path(value, base_dir=base_dir, root=root), extra))
        return specs

    participants = parse_participant_ids((config.get("participants", {}) or {}).get("ids"))
    template = dataset.get("participant_file") or dataset.get("file_template")
    specs = []
    for participant in participants:
        path = expand_path(str(template).format(participant=participant), base_dir=base_dir, root=root)
        specs.append((path, {"participant": participant}))
    return specs


def load_epoch_dataset_from_config(
    config_or_path: Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
    overrides: Sequence[str] | None = None,
    check_files: bool = False,
) -> EpochDataset:
    """Load a configured dataset into the neutral ``EpochDataset`` representation."""

    if isinstance(config_or_path, (str, Path)):
        config_path = Path(config_or_path)
        config = load_config(config_path)
        effective_base_dir = config_path.parent
    else:
        config = dict(config_or_path)
        effective_base_dir = Path(".") if base_dir is None else Path(base_dir)

    config = apply_overrides(config, overrides)
    validate_dataset_config(config, base_dir=effective_base_dir, check_files=check_files)
    dataset = _dataset_section(config)
    dataset_type = dataset["type"]

    if dataset_type == "mne_epochs":
        return _load_mne_epochs_dataset(config, base_dir=effective_base_dir)

    if dataset_type == "fieldtrip_mat":
        loader_config = {**dataset, "metadata": config.get("metadata", {}) or {}}
        if "validation" in config:
            loader_config["validation"] = config["validation"]
        loaded = [
            load_fieldtrip_mat_epochs(path, loader_config, extra_metadata=extra_metadata)
            for path, extra_metadata in _fieldtrip_file_specs(config, base_dir=effective_base_dir)
        ]
        name = str(dataset.get("name") or "fieldtrip_mat")
        return loaded[0] if len(loaded) == 1 else EpochDataset.concatenate(loaded, name=name)

    raise ConfigValidationError(f"Unsupported dataset.type: {dataset_type}")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a JSON/YAML dataset config."""

    parser = argparse.ArgumentParser(description="Validate a NeuRepTrace dataset config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-files", action="store_true", help="Require configured input files to exist.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key, e.g. --set participants.ids='[2,3]'.")
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.overrides)
    warnings = validate_dataset_config(config, base_dir=args.config.parent, check_files=args.check_files)
    print(f"Validated {args.config}")
    for warning in warnings:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
