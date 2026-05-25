"""Config loading and validation for dataset-driven NeuRepTrace workflows."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.io.dataset import EpochDataset
from neureptrace.io.fieldtrip_mat import load_fieldtrip_mat_epochs

SUPPORTED_DATASET_TYPES = {"fieldtrip_mat", "mne_epochs"}
DEFAULT_SCHEMA_VERSION = "neureptrace.dataset.v1"
CHANNEL_POLICIES = {"exact", "intersection", "first_dataset"}


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


def _normalize_path_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_path_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_path_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_path_for_json(item) for item in value]
    return value


def stable_config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON/YAML config mapping."""

    payload = json.dumps(_normalize_path_for_json(dict(config)), sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:" + sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 hash for a file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def effective_config(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    include_resolved_files: bool = True,
) -> dict[str, Any]:
    """Return a printable config with defaults, expanded participants, and resolved input files."""

    rendered = copy.deepcopy(dict(config))
    rendered.setdefault("schema_version", DEFAULT_SCHEMA_VERSION)
    participants = rendered.setdefault("participants", {})
    if isinstance(participants, dict) and "ids" in participants:
        participants["expanded_ids"] = parse_participant_ids(participants.get("ids"))
    if include_resolved_files:
        try:
            rendered["resolved_input_files"] = [str(path) for path in iter_dataset_files(rendered, base_dir=base_dir)]
        except Exception as exc:  # pragma: no cover - best-effort diagnostic rendering
            rendered["resolved_input_files_error"] = str(exc)
    rendered["effective_config_hash"] = stable_config_hash(rendered)
    return rendered


def _dataset_section(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ConfigValidationError("Config must contain a 'dataset' mapping.")
    return dataset


def _metadata_section(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = config.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        raise ConfigValidationError("Config section 'metadata' must be a mapping.")
    return metadata


def _validation_section(config: Mapping[str, Any]) -> dict[str, Any]:
    validation = config.get("validation", {}) or {}
    if not isinstance(validation, dict):
        raise ConfigValidationError("Config section 'validation' must be a mapping.")
    return validation


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
        if not (dataset.get("epochs") or dataset.get("epochs_file") or dataset.get("epochs_files")):
            raise ConfigValidationError("mne_epochs configs require dataset.epochs, dataset.epochs_file, or dataset.epochs_files.")
        epochs_files = dataset.get("epochs_files")
        if isinstance(epochs_files, Mapping):
            template = epochs_files.get("template") or epochs_files.get("path") or epochs_files.get("file")
            if template is None:
                raise ConfigValidationError("dataset.epochs_files mappings must contain template, path, or file.")
            if not parse_participant_ids((config.get("participants", {}) or {}).get("ids")):
                raise ConfigValidationError("mne_epochs dataset.epochs_files mappings require participants.ids.")
    if dataset_type == "fieldtrip_mat":
        participants = config.get("participants", {}) or {}
        has_participant_template = bool(
            dataset.get("participant_file")
            or dataset.get("file_template")
            or dataset.get("file_templates")
            or dataset.get("participant_files")
        )
        has_files = bool(dataset.get("files"))
        if not has_files and not has_participant_template:
            raise ConfigValidationError("fieldtrip_mat configs require dataset.files, dataset.file_templates, or dataset.participant_file.")
        if has_participant_template and not parse_participant_ids(participants.get("ids")):
            raise ConfigValidationError("fieldtrip_mat participant templates and file_templates require participants.ids.")

    metadata = _metadata_section(config)
    for column in metadata.get("columns", []) or []:
        if not isinstance(column, dict) or "name" not in column or "index" not in column:
            raise ConfigValidationError("metadata.columns entries must contain name and index.")
    for filter_spec in metadata.get("filters", []) or []:
        if not isinstance(filter_spec, dict) or "column" not in filter_spec:
            raise ConfigValidationError("metadata.filters entries must contain a column.")
        if "include" not in filter_spec and "exclude" not in filter_spec:
            raise ConfigValidationError("metadata.filters entries must contain include or exclude.")

    channel_policy = str(_validation_section(config).get("channel_policy", "exact"))
    if channel_policy not in CHANNEL_POLICIES:
        raise ConfigValidationError(f"validation.channel_policy must be one of {sorted(CHANNEL_POLICIES)}.")

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
        return [
            file_path
            for epochs_path, metadata_path in _mne_epoch_file_specs(config, base_dir=base_dir)
            for file_path in (epochs_path, metadata_path)
            if file_path is not None
        ]

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

    template_specs = _participant_file_templates(dataset)
    if template_specs:
        participants = parse_participant_ids((config.get("participants", {}) or {}).get("ids"))
        return [expand_path(template.format(**_format_values_for_participant(participant)), base_dir=base_dir, root=root) for participant in participants for _role, template, _extra in template_specs]

    participants = parse_participant_ids((config.get("participants", {}) or {}).get("ids"))
    template = dataset.get("participant_file") or dataset.get("file_template")
    return [expand_path(str(template).format(participant=participant), base_dir=base_dir, root=root) for participant in participants]


def _mne_epoch_file_specs(config: Mapping[str, Any], *, base_dir: str | Path) -> list[tuple[Path, Path | None]]:
    dataset = _dataset_section(config)
    root = dataset.get("root")
    epochs_files = dataset.get("epochs_files")
    if epochs_files is not None:
        if isinstance(epochs_files, Mapping):
            template = epochs_files.get("template") or epochs_files.get("path") or epochs_files.get("file")
            if template is None:
                raise ConfigValidationError("dataset.epochs_files mappings must contain template, path, or file.")
            participants = parse_participant_ids((config.get("participants", {}) or {}).get("ids"))
            return [
                (
                    expand_path(str(template).format(**_format_values_for_participant(participant)), base_dir=base_dir, root=root),
                    None,
                )
                for participant in participants
            ]
        if isinstance(epochs_files, str):
            return [(expand_path(epochs_files, base_dir=base_dir, root=root), None)]
        specs: list[tuple[Path, Path | None]] = []
        for item in epochs_files:
            if isinstance(item, Mapping):
                value = item.get("path") or item.get("file") or item.get("epochs")
                metadata_csv = item.get("metadata_csv")
            else:
                value = item
                metadata_csv = None
            if value is None:
                raise ConfigValidationError("dataset.epochs_files entries must contain a path.")
            specs.append(
                (
                    expand_path(str(value), base_dir=base_dir, root=root),
                    expand_path(str(metadata_csv), base_dir=base_dir, root=root) if metadata_csv else None,
                )
            )
        return specs

    epochs_path = expand_path(dataset.get("epochs") or dataset.get("epochs_file"), base_dir=base_dir, root=root)
    metadata_csv = dataset.get("metadata_csv")
    return [(epochs_path, expand_path(metadata_csv, base_dir=base_dir, root=root) if metadata_csv else None)]


def _load_single_mne_epochs_dataset(epochs_path: Path, metadata_csv: Path | None, *, name: str) -> EpochDataset:
    import mne

    epochs = mne.read_epochs(epochs_path, preload=True, verbose="error")
    metadata = epochs.metadata.copy() if epochs.metadata is not None else None
    if metadata_csv is not None:
        metadata = pd.read_csv(metadata_csv)
    if metadata is None:
        metadata = pd.DataFrame(index=range(len(epochs)))
    if len(metadata) != len(epochs):
        raise ValueError(f"Metadata row count ({len(metadata)}) does not match epochs ({len(epochs)}).")
    return EpochDataset(
        data=epochs.get_data(copy=True),
        times=epochs.times.copy(),
        channel_names=list(epochs.ch_names),
        metadata=metadata.reset_index(drop=True),
        name=name,
        provenance={"path": str(epochs_path), "loader": "mne_epochs"},
    )


def _load_mne_epochs_dataset(config: Mapping[str, Any], *, base_dir: str | Path) -> EpochDataset:
    dataset = _dataset_section(config)
    specs = _mne_epoch_file_specs(config, base_dir=base_dir)
    loaded = [
        _load_single_mne_epochs_dataset(path, metadata_path, name=path.stem)
        for path, metadata_path in specs
    ]
    if len(loaded) == 1:
        return loaded[0]
    name = str(dataset.get("name") or "mne_epochs")
    channel_policy = str(_validation_section(config).get("channel_policy", "exact"))
    return EpochDataset.concatenate(loaded, name=name, channel_policy=channel_policy)


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

    template_specs = _participant_file_templates(dataset)
    if template_specs:
        participants = parse_participant_ids((config.get("participants", {}) or {}).get("ids"))
        specs = []
        for participant in participants:
            format_values = _format_values_for_participant(participant)
            for _role, template, extra in template_specs:
                path = expand_path(template.format(**format_values), base_dir=base_dir, root=root)
                specs.append((path, {"participant": str(participant), **extra}))
        return specs

    participants = parse_participant_ids((config.get("participants", {}) or {}).get("ids"))
    template = dataset.get("participant_file") or dataset.get("file_template")
    specs = []
    for participant in participants:
        path = expand_path(str(template).format(participant=participant), base_dir=base_dir, root=root)
        specs.append((path, {"participant": participant}))
    return specs


def _format_values_for_participant(participant: int | str) -> dict[str, Any]:
    """Return path-template values for a participant token."""

    text = str(participant)
    values: dict[str, Any] = {
        "participant": text,
        "subject": text,
    }
    try:
        number = int(text)
    except ValueError:
        return values
    values.update(
        {
            "participant_int": number,
            "subject_int": number,
            "participant02d": f"{number:02d}",
            "subject02d": f"{number:02d}",
            "participant03d": f"{number:03d}",
            "subject03d": f"{number:03d}",
        }
    )
    return values


def _participant_file_templates(dataset: Mapping[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Return participant-expanded FieldTrip file templates with per-file metadata."""

    raw_templates = dataset.get("file_templates") or dataset.get("participant_files")
    if raw_templates is None:
        return []
    if not isinstance(raw_templates, Mapping):
        raise ConfigValidationError("dataset.file_templates must be a mapping from split/role name to path template.")

    specs: list[tuple[str, str, dict[str, Any]]] = []
    for role, value in raw_templates.items():
        if isinstance(value, Mapping):
            template = value.get("path") or value.get("file") or value.get("template")
            extra = {str(key): item for key, item in value.items() if key not in {"path", "file", "template"}}
        else:
            template = value
            extra = {}
        if template is None or str(template).strip() == "":
            raise ConfigValidationError(f"dataset.file_templates.{role} must contain a path template.")
        metadata = {"split": str(role), "file_role": str(role)}
        metadata.update(extra)
        specs.append((str(role), str(template), metadata))
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
        channel_policy = str(_validation_section(config).get("channel_policy", "exact"))
        return loaded[0] if len(loaded) == 1 else EpochDataset.concatenate(loaded, name=name, channel_policy=channel_policy)

    raise ConfigValidationError(f"Unsupported dataset.type: {dataset_type}")


def provenance_payload(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    base_dir: str | Path,
    include_file_hashes: bool = True,
) -> dict[str, Any]:
    """Build a reproducibility payload for config-driven outputs."""

    inputs = []
    for path in iter_dataset_files(config, base_dir=base_dir):
        entry = {"path": str(path)}
        if include_file_hashes and path.exists():
            entry["sha256"] = file_sha256(path)
        inputs.append(entry)
    return {
        "schema_version": config.get("schema_version", DEFAULT_SCHEMA_VERSION),
        "config_path": str(config_path),
        "effective_config_hash": stable_config_hash(config),
        "input_files": inputs,
        "random_seed": 13,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a JSON/YAML dataset config."""

    parser = argparse.ArgumentParser(description="Validate a NeuRepTrace dataset config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-files", action="store_true", help="Require configured input files to exist.")
    parser.add_argument("--print-effective-config", action="store_true", help="Print the config after overrides, defaults, participant expansion, and input-file resolution.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key, e.g. --set participants.ids='[2,3]'.")
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.overrides)
    warnings = validate_dataset_config(config, base_dir=args.config.parent, check_files=args.check_files)
    if args.print_effective_config:
        print(json.dumps(effective_config(config, base_dir=args.config.parent), indent=2, sort_keys=True, default=str))
    print(f"Validated {args.config}")
    for warning in warnings:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())