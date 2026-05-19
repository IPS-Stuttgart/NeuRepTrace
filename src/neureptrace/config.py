"""Typed YAML/JSON configuration for NeuRepTrace datasets and workflows.

The config layer is intentionally declarative. It describes dataset roots, file
roles, loaders, metadata files, participants, and workflow roles, but it does not
interpret modality-specific files. Loader implementations remain regular Python
code and can be supplied by NeuRepTrace, PyMEGDec, or another project package.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a NeuRepTrace config is structurally invalid."""


_RANGE_RE = re.compile(r"^(?P<start>\d+)\s*-\s*(?P<stop>\d+)$")


def parse_participants(value: str | Sequence[str | int] | None) -> tuple[str, ...]:
    """Parse a compact participant specification.

    Parameters
    ----------
    value:
        Either a comma-separated string such as ``"1-4,6,8"`` or an explicit
        sequence such as ``["sub-01", "sub-02"]``. Numeric ranges preserve zero
        padding when either endpoint is padded, for example ``"01-03"``.

    Returns
    -------
    tuple[str, ...]
        Participant identifiers in declaration order, with duplicates removed.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    elif isinstance(value, Sequence):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    else:
        raise ConfigError("participants must be a comma-separated string or a sequence")

    participants: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        match = _RANGE_RE.match(token)
        if match is None:
            expanded = [token]
        else:
            start_text = match.group("start")
            stop_text = match.group("stop")
            start = int(start_text)
            stop = int(stop_text)
            step = 1 if stop >= start else -1
            width = max(len(start_text), len(stop_text)) if start_text.startswith("0") or stop_text.startswith("0") else 0
            expanded = [f"{number:0{width}d}" if width else str(number) for number in range(start, stop + step, step)]
        for participant in expanded:
            if participant not in seen:
                participants.append(participant)
                seen.add(participant)
    return tuple(participants)


def _ensure_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_template(template: str, context: Mapping[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        name = exc.args[0]
        raise ConfigError(f"template uses unknown placeholder {{{name}}}: {template}") from exc


def _resolve_path(value: str, base_dir: Path, context: Mapping[str, Any] | None = None) -> Path:
    text = _format_template(value, context or {})
    text = os.path.expanduser(os.path.expandvars(text))
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _role_tuple(value: str | Sequence[str] | None, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence):
        roles = tuple(str(item).strip() for item in value if str(item).strip())
        return roles
    raise ConfigError(f"workflow.{field_name} must be a string or sequence of strings")


@dataclass(frozen=True)
class MetadataSpec:
    """Metadata columns and optional metadata table shared by dataset files."""

    path: str | None = None
    key: str | None = None
    label_column: str | None = None
    group_column: str | None = None
    trial_index_column: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MetadataSpec":
        if value is None:
            return cls()
        mapping = _ensure_mapping(value, "metadata")
        params = mapping.get("params", {}) or {}
        return cls(
            path=_optional_str(mapping, "path"),
            key=_optional_str(mapping, "key"),
            label_column=_optional_str(mapping, "label_column"),
            group_column=_optional_str(mapping, "group_column"),
            trial_index_column=_optional_str(mapping, "trial_index_column"),
            params=dict(_ensure_mapping(params, "metadata.params")),
        )

    def resolve_path(self, base_dir: Path, context: Mapping[str, Any] | None = None) -> Path | None:
        """Resolve the configured metadata table path, if present."""
        if self.path is None:
            return None
        return _resolve_path(self.path, base_dir, context)


@dataclass(frozen=True)
class FileSpec:
    """A named dataset file role, such as ``main`` or ``calibration``."""

    role: str
    loader: str
    path: str | None = None
    pattern: str | None = None
    metadata: str | None = None
    events: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, role: str, value: Mapping[str, Any]) -> "FileSpec":
        mapping = _ensure_mapping(value, f"dataset.files.{role}")
        path = _optional_str(mapping, "path")
        pattern = _optional_str(mapping, "pattern")
        loader = _optional_str(mapping, "loader")
        params = mapping.get("params", {}) or {}
        if path is None and pattern is None:
            raise ConfigError(f"dataset.files.{role} must define either path or pattern")
        if path is not None and pattern is not None:
            raise ConfigError(f"dataset.files.{role} must not define both path and pattern")
        if loader is None:
            raise ConfigError(f"dataset.files.{role}.loader is required")
        return cls(
            role=role,
            loader=loader,
            path=path,
            pattern=pattern,
            metadata=_optional_str(mapping, "metadata"),
            events=_optional_str(mapping, "events"),
            params=dict(_ensure_mapping(params, f"dataset.files.{role}.params")),
        )

    @property
    def template(self) -> str:
        """Return the path template backing this file role."""
        template = self.path if self.path is not None else self.pattern
        if template is None:  # Defensive: construction validates this.
            raise ConfigError(f"dataset.files.{self.role} has no path template")
        return template

    def uses_participant(self) -> bool:
        """Whether the file role references the ``{participant}`` placeholder."""
        templates = [self.template]
        if self.metadata is not None:
            templates.append(self.metadata)
        if self.events is not None:
            templates.append(self.events)
        return any("{participant" in template for template in templates)

    def resolve_path(self, base_dir: Path, context: Mapping[str, Any]) -> Path:
        """Resolve the data path for a concrete participant context."""
        return _resolve_path(self.template, base_dir, context)

    def resolve_metadata(self, base_dir: Path, context: Mapping[str, Any]) -> Path | None:
        """Resolve the role-specific metadata path, if configured."""
        if self.metadata is None:
            return None
        return _resolve_path(self.metadata, base_dir, context)

    def resolve_events(self, base_dir: Path, context: Mapping[str, Any]) -> Path | None:
        """Resolve the role-specific events path, if configured."""
        if self.events is None:
            return None
        return _resolve_path(self.events, base_dir, context)


@dataclass(frozen=True)
class MaterializedFile:
    """Concrete file path derived from a dataset spec."""

    dataset_id: str
    role: str
    participant: str | None
    loader: str
    path: Path
    metadata_path: Path | None = None
    events_path: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of dataset files and metadata."""

    id: str
    root: str | None = None
    root_env: str | None = None
    participants: tuple[str, ...] = ()
    files: dict[str, FileSpec] = field(default_factory=dict)
    metadata: MetadataSpec = field(default_factory=MetadataSpec)
    params: dict[str, Any] = field(default_factory=dict)
    base_dir: Path = field(default_factory=lambda: Path.cwd(), repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, base_dir: Path | None = None) -> "DatasetSpec":
        mapping = _ensure_mapping(value, "dataset")
        dataset_id = _optional_str(mapping, "id")
        if dataset_id is None:
            raise ConfigError("dataset.id is required")
        files_mapping = mapping.get("files", mapping.get("runs"))
        if files_mapping is None:
            raise ConfigError("dataset.files is required")
        files_source = _ensure_mapping(files_mapping, "dataset.files")
        files = {str(role): FileSpec.from_mapping(str(role), _ensure_mapping(spec, f"dataset.files.{role}")) for role, spec in files_source.items()}
        params = mapping.get("params", {}) or {}
        return cls(
            id=dataset_id,
            root=_optional_str(mapping, "root"),
            root_env=_optional_str(mapping, "root_env"),
            participants=parse_participants(mapping.get("participants")),
            files=files,
            metadata=MetadataSpec.from_mapping(mapping.get("metadata")),
            params=dict(_ensure_mapping(params, "dataset.params")),
            base_dir=base_dir or Path.cwd(),
        )

    def resolve_root(self) -> Path:
        """Resolve the dataset root relative to the config file."""
        if self.root is not None and self.root_env is not None:
            raise ConfigError("dataset.root and dataset.root_env are mutually exclusive")
        if self.root_env is not None:
            value = os.environ.get(self.root_env)
            if value is None or value.strip() == "":
                raise ConfigError(f"environment variable {self.root_env} is not set")
            return _resolve_path(value, self.base_dir)
        if self.root is not None:
            return _resolve_path(self.root, self.base_dir)
        return self.base_dir

    def context_for_participant(self, participant: str | None) -> dict[str, Any]:
        """Build a template context for one participant."""
        context: dict[str, Any] = {"dataset": self.id}
        if participant is not None:
            context["participant"] = participant
            try:
                context["participant_int"] = int(participant)
            except ValueError:
                pass
        return context

    def iter_files(self, roles: Sequence[str] | None = None) -> list[MaterializedFile]:
        """Materialize configured files for the requested roles."""
        selected_roles = tuple(roles) if roles is not None else tuple(self.files)
        root = self.resolve_root()
        materialized: list[MaterializedFile] = []
        for role in selected_roles:
            if role not in self.files:
                raise ConfigError(f"unknown file role: {role}")
            spec = self.files[role]
            participants: tuple[str | None, ...]
            participants = self.participants if spec.uses_participant() else (None,)
            for participant in participants:
                context = self.context_for_participant(participant)
                materialized.append(
                    MaterializedFile(
                        dataset_id=self.id,
                        role=role,
                        participant=participant,
                        loader=spec.loader,
                        path=spec.resolve_path(root, context),
                        metadata_path=spec.resolve_metadata(root, context),
                        events_path=spec.resolve_events(root, context),
                        params=dict(spec.params),
                    )
                )
        return materialized

    def validate(self) -> list[str]:
        """Return structural validation messages for this dataset spec."""
        messages: list[str] = []
        if self.root is not None and self.root_env is not None:
            messages.append("dataset.root and dataset.root_env are mutually exclusive")
        if not self.files:
            messages.append("dataset.files must contain at least one file role")
        for role, spec in self.files.items():
            if not role.strip():
                messages.append("dataset.files contains an empty role name")
            if spec.uses_participant() and not self.participants:
                messages.append(f"dataset.files.{role} uses {{participant}} but dataset.participants is empty")
        return messages


@dataclass(frozen=True)
class WorkflowSpec:
    """Declarative description of a decoding workflow over dataset file roles."""

    name: str
    train: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    label: str | None = None
    group: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkflowSpec":
        mapping = _ensure_mapping(value, "workflow")
        name = _optional_str(mapping, "name") or "default"
        params = mapping.get("params", {}) or {}
        return cls(
            name=name,
            train=_role_tuple(mapping.get("train"), field_name="train"),
            test=_role_tuple(mapping.get("test"), field_name="test"),
            label=_optional_str(mapping, "label"),
            group=_optional_str(mapping, "group"),
            params=dict(_ensure_mapping(params, "workflow.params")),
        )

    def referenced_roles(self) -> tuple[str, ...]:
        """Return all dataset roles referenced by the workflow."""
        return tuple(dict.fromkeys((*self.train, *self.test)))


@dataclass(frozen=True)
class ConfigSpec:
    """Top-level NeuRepTrace dataset/workflow configuration."""

    version: int
    dataset: DatasetSpec
    workflows: tuple[WorkflowSpec, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, base_dir: Path | None = None) -> "ConfigSpec":
        mapping = _ensure_mapping(value, "config")
        version = int(mapping.get("version", 1))
        if version != 1:
            raise ConfigError(f"unsupported config version: {version}")
        dataset = DatasetSpec.from_mapping(_ensure_mapping(mapping.get("dataset"), "dataset"), base_dir=base_dir)
        workflows = _workflows_from_mapping(mapping)
        return cls(version=version, dataset=dataset, workflows=workflows)

    def validate(self, *, check_paths: bool = False) -> list[str]:
        """Return validation messages for the complete configuration."""
        messages = self.dataset.validate()
        for workflow in self.workflows:
            for role in workflow.referenced_roles():
                if role not in self.dataset.files:
                    messages.append(f"workflow.{workflow.name} references unknown dataset file role: {role}")
            if workflow.label is None and self.dataset.metadata.label_column is None:
                messages.append(f"workflow.{workflow.name} does not define label and dataset.metadata.label_column is empty")
        if check_paths:
            messages.extend(self._validate_paths())
        return messages

    def _validate_paths(self) -> list[str]:
        messages: list[str] = []
        try:
            files = self.dataset.iter_files()
        except ConfigError as exc:
            return [str(exc)]
        for item in files:
            if not item.path.exists():
                messages.append(f"{item.role}: data file does not exist: {item.path}")
            if item.metadata_path is not None and not item.metadata_path.exists():
                messages.append(f"{item.role}: metadata file does not exist: {item.metadata_path}")
            if item.events_path is not None and not item.events_path.exists():
                messages.append(f"{item.role}: events file does not exist: {item.events_path}")
        dataset_metadata = self.dataset.metadata.resolve_path(self.dataset.resolve_root())
        if dataset_metadata is not None and not dataset_metadata.exists():
            messages.append(f"dataset metadata file does not exist: {dataset_metadata}")
        return messages


def _workflows_from_mapping(mapping: Mapping[str, Any]) -> tuple[WorkflowSpec, ...]:
    has_workflow = "workflow" in mapping and mapping["workflow"] is not None
    has_workflows = "workflows" in mapping and mapping["workflows"] is not None
    if has_workflow and has_workflows:
        raise ConfigError("use either workflow or workflows, not both")
    if has_workflow:
        return (WorkflowSpec.from_mapping(_ensure_mapping(mapping["workflow"], "workflow")),)
    if not has_workflows:
        return ()
    source = mapping["workflows"]
    if isinstance(source, Mapping):
        return tuple(WorkflowSpec.from_mapping({**_ensure_mapping(spec, f"workflows.{name}"), "name": name}) for name, spec in source.items())
    if isinstance(source, Sequence) and not isinstance(source, str):
        return tuple(WorkflowSpec.from_mapping(_ensure_mapping(item, "workflows[]")) for item in source)
    raise ConfigError("workflows must be a mapping or sequence")


def load_config(path: str | Path) -> ConfigSpec:
    """Load a NeuRepTrace config from a JSON, YAML, or YML file."""
    config_path = Path(path)
    data = _load_mapping_file(config_path)
    return ConfigSpec.from_mapping(data, base_dir=config_path.parent)


def _load_mapping_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = yaml.safe_load(text)
    if data is None:
        data = {}
    return _ensure_mapping(data, str(path))


def _report_payload(config: ConfigSpec, messages: Sequence[str], *, list_files: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": not messages,
        "dataset": config.dataset.id,
        "workflows": [workflow.name for workflow in config.workflows],
        "messages": list(messages),
    }
    if list_files:
        payload["files"] = [
            {
                "role": item.role,
                "participant": item.participant,
                "loader": item.loader,
                "path": str(item.path),
                "metadata_path": None if item.metadata_path is None else str(item.metadata_path),
                "events_path": None if item.events_path is None else str(item.events_path),
            }
            for item in config.dataset.iter_files()
        ]
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Validate a NeuRepTrace YAML/JSON dataset configuration."""
    parser = argparse.ArgumentParser(description="Validate a NeuRepTrace YAML/JSON dataset configuration.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-files", action="store_true", help="also verify that materialized data, metadata, and events files exist")
    parser.add_argument("--list-files", action="store_true", help="print materialized dataset files")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit a machine-readable validation report")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        messages = config.validate(check_paths=args.check_files)
        payload = _report_payload(config, messages, list_files=args.list_files)
    except ConfigError as exc:
        payload = {"ok": False, "dataset": None, "workflows": [], "messages": [str(exc)]}

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        status = "ok" if payload["ok"] else "error"
        print(f"{status}\tdataset={payload['dataset']}")
        for message in payload["messages"]:
            print(f"error\t{message}")
        if args.list_files and payload["ok"]:
            for item in payload.get("files", []):
                participant = "" if item["participant"] is None else f"\tparticipant={item['participant']}"
                print(f"file\trole={item['role']}{participant}\tloader={item['loader']}\tpath={item['path']}")
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
