"""Declarative workflow specifications for dataset-level NeuRepTrace runs.

The workflow-spec layer describes *what* should be loaded and evaluated without
hard-coding project conventions in NeuRepTrace itself.  Dataset packages such as
PyMEGDec can register or document loader names, while this module provides the
stable, upstream shape for configuration files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_JSON_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"


class WorkflowSpecError(ValueError):
    """Raised when a workflow configuration is structurally invalid."""


@dataclass(frozen=True)
class WorkflowDefinition:
    """Top-level workflow identity.

    Parameters
    ----------
    kind
        Stable workflow kind, for example ``cross_subject_decoding`` or
        ``mne_time_decode``.  Runtime dispatchers can use this value later.
    name
        Optional human-readable experiment name.
    options
        Additional workflow-specific keys preserved for future runners.
    """

    kind: str
    name: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetFileSpec:
    """A data source role within a dataset specification."""

    role: str
    loader: str
    path: str | None = None
    pattern: str | None = None
    metadata: str | None = None
    format: str | None = None
    required: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetadataFileSpec:
    """Metadata table associated with a dataset or a data-file role."""

    role: str
    path: str
    key: str | None = None
    columns: list[str] | None = None
    required: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset declaration independent of project-specific loading code."""

    id: str
    root: str | None = None
    participants: Any = None
    files: dict[str, DatasetFileSpec] = field(default_factory=dict)
    metadata: dict[str, MetadataFileSpec] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Model/classifier declaration."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation split and role declaration."""

    scheme: str
    train: str | None = None
    test: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputsSpec:
    """Named output paths for a workflow."""

    root: str | None = None
    tables: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    """Normalized representation of a NeuRepTrace workflow config."""

    version: int
    workflow: WorkflowDefinition
    dataset: DatasetSpec
    features: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    normalization: Any = None
    model: ModelSpec | None = None
    evaluation: EvaluationSpec | None = None
    outputs: OutputsSpec | None = None
    options: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None


def _expect_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowSpecError(f"{context} must be a mapping/object")
    return dict(value)


def _optional_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _expect_mapping(value, context)


def _required_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise WorkflowSpecError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, context)


def _bool_value(value: Any, context: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise WorkflowSpecError(f"{context} must be a boolean")


def _string_list(value: Any, context: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_required_string(item, f"{context}[]") for item in value]
    raise WorkflowSpecError(f"{context} must be a string or a list of strings")


def _extra_options(block: Mapping[str, Any], known: Iterable[str]) -> dict[str, Any]:
    known_set = set(known)
    options = _optional_mapping(block.get("options"), "options")
    for key, value in block.items():
        if key not in known_set and key != "options":
            options[key] = value
    return options


def _parse_workflow(value: Any) -> WorkflowDefinition:
    if isinstance(value, str):
        return WorkflowDefinition(kind=_required_string(value, "workflow"))
    block = _expect_mapping(value, "workflow")
    kind = block.get("kind", block.get("type", block.get("workflow")))
    name = _optional_string(block.get("name"), "workflow.name")
    if kind is None and name is not None:
        kind = name
    return WorkflowDefinition(
        kind=_required_string(kind, "workflow.kind"),
        name=name,
        options=_extra_options(block, {"kind", "type", "workflow", "name"}),
    )


def _parse_file_spec(role: str, value: Any) -> DatasetFileSpec:
    block = _expect_mapping(value, f"dataset.files.{role}")
    loader = _required_string(block.get("loader"), f"dataset.files.{role}.loader")
    path = _optional_string(block.get("path"), f"dataset.files.{role}.path")
    pattern = _optional_string(block.get("pattern"), f"dataset.files.{role}.pattern")
    if path is None and pattern is None:
        raise WorkflowSpecError(f"dataset.files.{role} must define either path or pattern")
    return DatasetFileSpec(
        role=role,
        loader=loader,
        path=path,
        pattern=pattern,
        metadata=_optional_string(block.get("metadata"), f"dataset.files.{role}.metadata"),
        format=_optional_string(block.get("format"), f"dataset.files.{role}.format"),
        required=_bool_value(block.get("required"), f"dataset.files.{role}.required", default=True),
        options=_extra_options(block, {"loader", "path", "pattern", "metadata", "format", "required"}),
    )


def _parse_metadata_spec(role: str, value: Any) -> MetadataFileSpec:
    block = _expect_mapping(value, f"dataset.metadata.{role}")
    return MetadataFileSpec(
        role=role,
        path=_required_string(block.get("path"), f"dataset.metadata.{role}.path"),
        key=_optional_string(block.get("key"), f"dataset.metadata.{role}.key"),
        columns=_string_list(block.get("columns"), f"dataset.metadata.{role}.columns"),
        required=_bool_value(block.get("required"), f"dataset.metadata.{role}.required", default=True),
        options=_extra_options(block, {"path", "key", "columns", "required"}),
    )


def _parse_dataset(value: Any) -> DatasetSpec:
    block = _expect_mapping(value, "dataset")
    file_blocks = _expect_mapping(block.get("files"), "dataset.files")
    if not file_blocks:
        raise WorkflowSpecError("dataset.files must define at least one data-file role")
    metadata_blocks = _optional_mapping(block.get("metadata", block.get("meta")), "dataset.metadata")
    return DatasetSpec(
        id=_required_string(block.get("id"), "dataset.id"),
        root=_optional_string(block.get("root"), "dataset.root"),
        participants=block.get("participants"),
        files={str(role): _parse_file_spec(str(role), spec) for role, spec in file_blocks.items()},
        metadata={str(role): _parse_metadata_spec(str(role), spec) for role, spec in metadata_blocks.items()},
        options=_extra_options(block, {"id", "root", "participants", "files", "metadata", "meta"}),
    )


def _parse_model(value: Any) -> ModelSpec | None:
    if value is None:
        return None
    if isinstance(value, str):
        return ModelSpec(name=_required_string(value, "model"))
    block = _expect_mapping(value, "model")
    name = block.get("name", block.get("classifier", block.get("estimator")))
    params = _optional_mapping(block.get("params"), "model.params")
    return ModelSpec(
        name=_required_string(name, "model.name"),
        params=params,
        options=_extra_options(block, {"name", "classifier", "estimator", "params"}),
    )


def _parse_evaluation(value: Any) -> EvaluationSpec | None:
    if value is None:
        return None
    if isinstance(value, str):
        return EvaluationSpec(scheme=_required_string(value, "evaluation"))
    block = _expect_mapping(value, "evaluation")
    scheme = block.get("scheme", block.get("split", block.get("kind")))
    return EvaluationSpec(
        scheme=_required_string(scheme, "evaluation.scheme"),
        train=_optional_string(block.get("train"), "evaluation.train"),
        test=_optional_string(block.get("test"), "evaluation.test"),
        options=_extra_options(block, {"scheme", "split", "kind", "train", "test"}),
    )


def _parse_outputs(value: Any) -> OutputsSpec | None:
    if value is None:
        return None
    block = _expect_mapping(value, "outputs")
    tables = _optional_mapping(block.get("tables"), "outputs.tables")
    for name, path in tables.items():
        _required_string(name, "outputs.tables key")
        _required_string(path, f"outputs.tables.{name}")
    return OutputsSpec(
        root=_optional_string(block.get("root"), "outputs.root"),
        tables={str(key): str(value) for key, value in tables.items()},
        options=_extra_options(block, {"root", "tables"}),
    )


def parse_workflow_spec(mapping: Mapping[str, Any], *, source: Path | None = None) -> WorkflowSpec:
    """Normalize and validate a workflow mapping.

    The parser intentionally validates only portable structure.  It does not try
    to import loaders or execute a workflow; those checks belong to runners and
    dataset packages.
    """
    block = _expect_mapping(mapping, "workflow specification")
    version = block.get("version")
    if not isinstance(version, int):
        raise WorkflowSpecError("version must be an integer")
    if version != 1:
        raise WorkflowSpecError(f"unsupported workflow spec version {version}; expected 1")

    return WorkflowSpec(
        version=version,
        workflow=_parse_workflow(block.get("workflow")),
        dataset=_parse_dataset(block.get("dataset")),
        features=_optional_mapping(block.get("features"), "features"),
        preprocessing=_optional_mapping(block.get("preprocessing"), "preprocessing"),
        normalization=block.get("normalization"),
        model=_parse_model(block.get("model")),
        evaluation=_parse_evaluation(block.get("evaluation")),
        outputs=_parse_outputs(block.get("outputs")),
        options=_extra_options(
            block,
            {"version", "workflow", "dataset", "features", "preprocessing", "normalization", "model", "evaluation", "outputs"},
        ),
        source=source,
    )


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional environment state.
        raise WorkflowSpecError("YAML workflow specs require PyYAML; install pyyaml or use .json") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_workflow_mapping(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML workflow config into a plain mapping."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    elif suffix in {".yml", ".yaml"}:
        loaded = _load_yaml(config_path)
    else:
        raise WorkflowSpecError(f"unsupported workflow spec extension '{config_path.suffix}'; use .json, .yml, or .yaml")
    return _expect_mapping(loaded, "workflow specification")


def load_workflow_spec(path: str | Path) -> WorkflowSpec:
    """Load and normalize a JSON/YAML workflow configuration."""
    config_path = Path(path)
    return parse_workflow_spec(load_workflow_mapping(config_path), source=config_path)


def workflow_spec_to_dict(spec: WorkflowSpec) -> dict[str, Any]:
    """Return a JSON-serializable normalized representation of a workflow spec."""
    data = asdict(spec)
    data.pop("source", None)
    return data


def workflow_json_schema() -> dict[str, Any]:
    """Return a compact JSON Schema for editor/tooling integration."""
    string_or_null = {"type": ["string", "null"]}
    string_map = {"type": "object", "additionalProperties": {"type": "string"}}
    file_spec = {
        "type": "object",
        "required": ["loader"],
        "anyOf": [{"required": ["path"]}, {"required": ["pattern"]}],
        "properties": {
            "loader": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "pattern": {"type": "string", "minLength": 1},
            "metadata": string_or_null,
            "format": string_or_null,
            "required": {"type": "boolean"},
            "options": {"type": "object"},
        },
        "additionalProperties": True,
    }
    metadata_spec = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "key": string_or_null,
            "columns": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
            "required": {"type": "boolean"},
            "options": {"type": "object"},
        },
        "additionalProperties": True,
    }
    return {
        "$schema": _JSON_SCHEMA_URI,
        "title": "NeuRepTrace workflow specification",
        "type": "object",
        "required": ["version", "workflow", "dataset"],
        "properties": {
            "version": {"const": 1},
            "workflow": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {
                        "type": "object",
                        "anyOf": [{"required": ["kind"]}, {"required": ["type"]}, {"required": ["name"]}],
                        "properties": {
                            "kind": {"type": "string", "minLength": 1},
                            "type": {"type": "string", "minLength": 1},
                            "name": {"type": "string", "minLength": 1},
                            "options": {"type": "object"},
                        },
                        "additionalProperties": True,
                    },
                ]
            },
            "dataset": {
                "type": "object",
                "required": ["id", "files"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "root": string_or_null,
                    "participants": {},
                    "files": {"type": "object", "minProperties": 1, "additionalProperties": file_spec},
                    "metadata": {"type": "object", "additionalProperties": metadata_spec},
                    "meta": {"type": "object", "additionalProperties": metadata_spec},
                    "options": {"type": "object"},
                },
                "additionalProperties": True,
            },
            "features": {"type": "object"},
            "preprocessing": {"type": "object"},
            "normalization": {},
            "model": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {
                        "type": "object",
                        "anyOf": [{"required": ["name"]}, {"required": ["classifier"]}, {"required": ["estimator"]}],
                        "properties": {"name": string_or_null, "classifier": string_or_null, "estimator": string_or_null, "params": {"type": "object"}},
                        "additionalProperties": True,
                    },
                ]
            },
            "evaluation": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {
                        "type": "object",
                        "anyOf": [{"required": ["scheme"]}, {"required": ["split"]}, {"required": ["kind"]}],
                        "properties": {"scheme": string_or_null, "split": string_or_null, "kind": string_or_null, "train": string_or_null, "test": string_or_null},
                        "additionalProperties": True,
                    },
                ]
            },
            "outputs": {"type": "object", "properties": {"root": string_or_null, "tables": string_map}, "additionalProperties": True},
        },
        "additionalProperties": True,
    }


def _expand_user_env(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value)))


def _resolve_path(value: str, *, base_dir: Path, root: str | None) -> Path:
    path = _expand_user_env(value)
    if path.is_absolute():
        return path
    if root is not None:
        root_path = _expand_user_env(root)
        if not root_path.is_absolute():
            root_path = base_dir / root_path
        return root_path / path
    return base_dir / path


def _participant_records(participants: Any) -> list[dict[str, Any]]:
    if not isinstance(participants, Sequence) or isinstance(participants, (bytes, bytearray, str)):
        return []
    records: list[dict[str, Any]] = []
    for participant in participants:
        if isinstance(participant, Mapping):
            record = dict(participant)
            identifier = record.get("participant", record.get("subject", record.get("id")))
            if identifier is not None:
                record.setdefault("participant", identifier)
                record.setdefault("subject", identifier)
                record.setdefault("id", identifier)
        else:
            record = {"participant": participant, "subject": participant, "id": participant}
        records.append(record)
    return records


def _format_pattern(pattern: str, record: Mapping[str, Any], context: str) -> str:
    try:
        return pattern.format(**record)
    except Exception as exc:  # pragma: no cover - Python string formatting can raise several concrete exceptions.
        raise WorkflowSpecError(f"could not format {context} with participant record {record}: {exc}") from exc


def check_workflow_files(spec: WorkflowSpec) -> list[str]:
    """Return path-related validation errors without importing dataset loaders."""
    base_dir = spec.source.parent if spec.source is not None else Path.cwd()
    root = spec.dataset.root
    messages: list[str] = []

    if root is not None:
        root_path = _resolve_path(".", base_dir=base_dir, root=root).resolve()
        if not root_path.exists():
            messages.append(f"dataset.root does not exist: {root_path}")

    participants = _participant_records(spec.dataset.participants)
    for role, file_spec in spec.dataset.files.items():
        if file_spec.path is not None:
            path = _resolve_path(file_spec.path, base_dir=base_dir, root=root)
            if file_spec.required and not path.exists():
                messages.append(f"dataset.files.{role}.path does not exist: {path}")
        elif file_spec.pattern is not None:
            if not participants:
                messages.append(f"dataset.files.{role}.pattern cannot be checked without a list-valued dataset.participants")
                continue
            missing = []
            for participant in participants:
                formatted = _format_pattern(file_spec.pattern, participant, f"dataset.files.{role}.pattern")
                path = _resolve_path(formatted, base_dir=base_dir, root=root)
                if file_spec.required and not path.exists():
                    missing.append(path)
            if missing:
                preview = ", ".join(str(path) for path in missing[:3])
                suffix = "" if len(missing) <= 3 else f" and {len(missing) - 3} more"
                messages.append(f"dataset.files.{role}.pattern has {len(missing)} missing file(s): {preview}{suffix}")

    for role, metadata_spec in spec.dataset.metadata.items():
        path = _resolve_path(metadata_spec.path, base_dir=base_dir, root=root)
        if metadata_spec.required and not path.exists():
            messages.append(f"dataset.metadata.{role}.path does not exist: {path}")

    return messages
