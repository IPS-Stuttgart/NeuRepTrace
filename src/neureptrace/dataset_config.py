"""YAML/JSON dataset configuration helpers for NeuRepTrace.

The module provides a small, loader-neutral manifest layer.  It validates dataset
and workflow references, resolves participant file patterns, and can compile an
MNE-Epochs decoding workflow into the CSV manifest consumed by
:mod:`neureptrace.benchmark`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

YamlJson = Mapping[str, Any]
ConfigSeverity = Literal["error", "warning"]

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")
_RANGE_PATTERN = re.compile(r"^(?P<start>\d+)\s*-\s*(?P<stop>\d+)$")
_MNE_EPOCHS_LOADERS = {"mne_epochs", "mne-epochs", "mne_epochs_fif", "epochs_fif"}
_BENCHMARK_OPTION_COLUMNS = {
    "picks",
    "tmin",
    "tmax",
    "window_ms",
    "step_ms",
    "n_splits",
    "max_iter",
    "decoder",
    "emission_mode",
    "feature_preprocessor",
    "pca_components",
    "tune_hyperparameters",
    "tuning_cv_splits",
    "tuning_scoring",
    "tuning_c_grid",
    "temporal_train_window",
    "temporal_train_window_start",
    "temporal_train_window_stop",
    "calibration_bins",
    "variant",
    "out_csv",
    "calibration_out_csv",
    "observation_out_csv",
    "metadata_out",
    "case_sensitive",
}
_METADATA_COLUMNS = {
    "label_column",
    "group_column",
    "source_column",
    "positive_pattern",
    "negative_pattern",
    "positive_label",
    "negative_label",
}


class DatasetConfigError(ValueError):
    """Raised when a YAML/JSON dataset configuration is structurally invalid."""


@dataclass(frozen=True)
class FileSpec:
    """One named file role in a dataset configuration."""

    role: str
    pattern: str
    loader: str = "mne_epochs"
    root: str | None = None
    required: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-level configuration shared by one or more workflows."""

    id: str
    root: str | None
    participants: tuple[str, ...]
    files: Mapping[str, FileSpec]
    subject_template: str = "{participant}"
    defaults: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    """A named analysis workflow bound to dataset file roles."""

    name: str
    kind: str = "mne_time_decode"
    epochs: str | None = None
    metadata: str | None = None
    events: str | None = None
    train: str | None = None
    test: str | None = None
    label_column: str | None = None
    group_column: str | None = None
    source_column: str | None = None
    positive_pattern: str | None = None
    negative_pattern: str | None = None
    positive_label: str | None = None
    negative_label: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig:
    """Parsed YAML/JSON dataset configuration."""

    version: int
    dataset: DatasetSpec
    workflows: tuple[WorkflowSpec, ...]
    config_dir: Path
    path: Path | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedFile:
    """A concrete participant/file-role path resolved from a pattern."""

    dataset_id: str
    participant: str
    subject: str
    role: str
    loader: str
    path: Path
    required: bool
    exists: bool
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationMessage:
    """One validation message emitted for a dataset configuration."""

    severity: ConfigSeverity
    message: str


@dataclass(frozen=True)
class ConfigValidation:
    """Validation result for a dataset configuration."""

    messages: tuple[ValidationMessage, ...]

    @property
    def ok(self) -> bool:
        """Return ``True`` when no validation error was emitted."""

        return not any(message.severity == "error" for message in self.messages)

    @property
    def errors(self) -> tuple[str, ...]:
        """Return error messages only."""

        return tuple(message.message for message in self.messages if message.severity == "error")

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return warning messages only."""

        return tuple(message.message for message in self.messages if message.severity == "warning")


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetConfigError(f"{label} must be a mapping/object.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _expand_env_vars(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise DatasetConfigError(f"Environment variable {name!r} is required by the dataset config but is not set.")

    return _ENV_PATTERN.sub(replace, value)


def _resolve_root(value: str | None, *, config_dir: Path) -> Path:
    if value is None:
        return config_dir
    expanded = Path(_expand_env_vars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = config_dir / expanded
    return expanded


def _expand_range_token(token: str) -> tuple[str, ...]:
    match = _RANGE_PATTERN.match(token)
    if match is None:
        return (token,)
    start_text = match.group("start")
    stop_text = match.group("stop")
    start = int(start_text)
    stop = int(stop_text)
    if stop < start:
        raise DatasetConfigError(f"Participant range {token!r} is descending; use an explicit list if that is intended.")
    width = max(len(start_text), len(stop_text)) if start_text.startswith("0") or stop_text.startswith("0") else 0
    if width:
        return tuple(f"{value:0{width}d}" for value in range(start, stop + 1))
    return tuple(str(value) for value in range(start, stop + 1))


def expand_participants(value: Any) -> tuple[str, ...]:
    """Expand a participant list or comma-separated inclusive ranges.

    Examples
    --------
    ``"1-3,6"`` becomes ``("1", "2", "3", "6")`` and ``"01-03"`` keeps
    two-digit zero padding.
    """

    if value is None:
        raise DatasetConfigError("dataset.participants is required.")
    if isinstance(value, str):
        tokens: Iterable[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        tokens = value
    else:
        raise DatasetConfigError("dataset.participants must be a list or comma-separated string.")

    participants: list[str] = []
    for token_value in tokens:
        token = str(token_value).strip()
        if not token:
            continue
        participants.extend(_expand_range_token(token))
    if not participants:
        raise DatasetConfigError("dataset.participants expanded to an empty list.")
    return tuple(participants)


def _file_spec_from_mapping(role: str, value: Any) -> FileSpec:
    if isinstance(value, str):
        return FileSpec(role=role, pattern=value)
    spec = _as_mapping(value, label=f"dataset.files.{role}")
    pattern = _optional_str(spec.get("pattern"))
    if pattern is None:
        raise DatasetConfigError(f"dataset.files.{role}.pattern is required.")
    reserved = {"pattern", "loader", "root", "required"}
    return FileSpec(
        role=role,
        pattern=pattern,
        loader=_optional_str(spec.get("loader")) or "mne_epochs",
        root=_optional_str(spec.get("root")),
        required=bool(spec.get("required", True)),
        options={key: value for key, value in spec.items() if key not in reserved},
    )


def _dataset_from_mapping(value: Mapping[str, Any]) -> DatasetSpec:
    root = _optional_str(value.get("root"))
    root_env = _optional_str(value.get("root_env"))
    if root is None and root_env is not None:
        root = "${" + root_env + "}"
    files_value = _as_mapping(value.get("files"), label="dataset.files")
    files = {str(role): _file_spec_from_mapping(str(role), spec) for role, spec in files_value.items()}
    if not files:
        raise DatasetConfigError("dataset.files must contain at least one file role.")
    defaults = value.get("defaults", {})
    return DatasetSpec(
        id=_optional_str(value.get("id")) or "dataset",
        root=root,
        participants=expand_participants(value.get("participants")),
        files=files,
        subject_template=_optional_str(value.get("subject_template")) or "{participant}",
        defaults=_as_mapping(defaults, label="dataset.defaults"),
    )


def _workflow_from_mapping(value: Mapping[str, Any], *, index: int) -> WorkflowSpec:
    reserved = {
        "name",
        "kind",
        "type",
        "epochs",
        "metadata",
        "events",
        "train",
        "test",
        "label",
        "label_column",
        "group",
        "group_column",
        "source_column",
        "positive_pattern",
        "negative_pattern",
        "positive_label",
        "negative_label",
        "options",
    }
    explicit_options = value.get("options", {})
    options = dict(_as_mapping(explicit_options, label=f"workflow[{index}].options"))
    for key, item in value.items():
        if key not in reserved and key in _BENCHMARK_OPTION_COLUMNS:
            options[key] = item
    return WorkflowSpec(
        name=_optional_str(value.get("name")) or f"workflow_{index + 1}",
        kind=_optional_str(value.get("kind")) or _optional_str(value.get("type")) or "mne_time_decode",
        epochs=_optional_str(value.get("epochs")),
        metadata=_optional_str(value.get("metadata")),
        events=_optional_str(value.get("events")),
        train=_optional_str(value.get("train")),
        test=_optional_str(value.get("test")),
        label_column=_optional_str(value.get("label_column")) or _optional_str(value.get("label")),
        group_column=_optional_str(value.get("group_column")) or _optional_str(value.get("group")),
        source_column=_optional_str(value.get("source_column")),
        positive_pattern=_optional_str(value.get("positive_pattern")),
        negative_pattern=_optional_str(value.get("negative_pattern")),
        positive_label=_optional_str(value.get("positive_label")),
        negative_label=_optional_str(value.get("negative_label")),
        options=options,
    )


def _workflows_from_mapping(config: Mapping[str, Any]) -> tuple[WorkflowSpec, ...]:
    if "workflows" in config:
        workflows_value = config["workflows"]
        if not isinstance(workflows_value, Sequence) or isinstance(workflows_value, (str, bytes)):
            raise DatasetConfigError("workflows must be a list.")
        workflows = tuple(_workflow_from_mapping(_as_mapping(value, label=f"workflows[{index}]"), index=index) for index, value in enumerate(workflows_value))
    elif "workflow" in config:
        workflows = (_workflow_from_mapping(_as_mapping(config["workflow"], label="workflow"), index=0),)
    else:
        workflows = ()
    return workflows


def dataset_config_from_mapping(config: Mapping[str, Any], *, config_dir: Path | str = ".", path: Path | None = None) -> DatasetConfig:
    """Create a :class:`DatasetConfig` from an already parsed mapping."""

    dataset = _dataset_from_mapping(_as_mapping(config.get("dataset"), label="dataset"))
    version = int(config.get("version", 1))
    workflows = _workflows_from_mapping(config)
    return DatasetConfig(version=version, dataset=dataset, workflows=workflows, config_dir=Path(config_dir), path=path, raw=dict(config))


def _read_yaml_json(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the optional dependency.
            raise DatasetConfigError("YAML config files require PyYAML. Install it or use JSON.") from exc
        data = yaml.safe_load(text)
    else:
        raise DatasetConfigError(f"Unsupported config suffix {suffix!r}; use .yml, .yaml, or .json.")
    return _as_mapping(data, label=str(path))


def load_dataset_config(path: Path | str) -> DatasetConfig:
    """Load a YAML/JSON dataset configuration."""

    config_path = Path(path)
    return dataset_config_from_mapping(_read_yaml_json(config_path), config_dir=config_path.parent, path=config_path)


def _subject_for_participant(config: DatasetConfig, participant: str) -> str:
    context = {"participant": participant, "participant_int": int(participant) if participant.isdigit() else participant, "dataset": config.dataset.id}
    try:
        return config.dataset.subject_template.format(**context)
    except KeyError as exc:
        raise DatasetConfigError(f"Unknown placeholder {exc.args[0]!r} in dataset.subject_template.") from exc


def _format_path_pattern(config: DatasetConfig, file_spec: FileSpec, *, participant: str, subject: str, workflow_name: str | None = None) -> str:
    context = {
        "dataset": config.dataset.id,
        "participant": participant,
        "participant_int": int(participant) if participant.isdigit() else participant,
        "subject": subject,
        "role": file_spec.role,
        "workflow": workflow_name or "",
    }
    try:
        return file_spec.pattern.format(**context)
    except KeyError as exc:
        raise DatasetConfigError(f"Unknown placeholder {exc.args[0]!r} in pattern for file role {file_spec.role!r}.") from exc


def resolve_file_path(config: DatasetConfig, file_spec: FileSpec, *, participant: str, subject: str, workflow_name: str | None = None) -> Path:
    """Resolve one file-role path for one participant."""

    root = _resolve_root(file_spec.root or config.dataset.root, config_dir=config.config_dir)
    pattern = _format_path_pattern(config, file_spec, participant=participant, subject=subject, workflow_name=workflow_name)
    path = Path(_expand_env_vars(pattern)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def resolve_files(config: DatasetConfig, *, workflow_name: str | None = None) -> list[ResolvedFile]:
    """Resolve all configured file-role paths for all participants."""

    rows: list[ResolvedFile] = []
    for participant in config.dataset.participants:
        subject = _subject_for_participant(config, participant)
        for file_spec in config.dataset.files.values():
            path = resolve_file_path(config, file_spec, participant=participant, subject=subject, workflow_name=workflow_name)
            rows.append(
                ResolvedFile(
                    dataset_id=config.dataset.id,
                    participant=participant,
                    subject=subject,
                    role=file_spec.role,
                    loader=file_spec.loader,
                    path=path,
                    required=file_spec.required,
                    exists=path.exists(),
                    options=file_spec.options,
                )
            )
    return rows


def _referenced_roles(workflow: WorkflowSpec) -> dict[str, str]:
    references = {
        "epochs": workflow.epochs,
        "metadata": workflow.metadata,
        "events": workflow.events,
        "train": workflow.train,
        "test": workflow.test,
    }
    return {name: role for name, role in references.items() if role is not None}


def validate_dataset_config(config: DatasetConfig, *, check_files: bool = False) -> ConfigValidation:
    """Validate structural references and, optionally, resolved file existence."""

    messages: list[ValidationMessage] = []
    if config.version != 1:
        messages.append(ValidationMessage("error", f"Unsupported config version {config.version}; expected version 1."))
    if not config.workflows:
        messages.append(ValidationMessage("warning", "No workflow is defined; only dataset file paths can be resolved."))

    roles = set(config.dataset.files)
    for workflow in config.workflows:
        for field_name, role in _referenced_roles(workflow).items():
            if role not in roles:
                messages.append(ValidationMessage("error", f"Workflow {workflow.name!r} references unknown {field_name} role {role!r}."))
        if workflow.kind == "mne_time_decode":
            epochs_role = workflow.epochs or ("epochs" if "epochs" in roles else workflow.train)
            if epochs_role is None:
                messages.append(ValidationMessage("error", f"Workflow {workflow.name!r} is mne_time_decode but has no epochs role."))
            elif epochs_role in config.dataset.files and config.dataset.files[epochs_role].loader not in _MNE_EPOCHS_LOADERS:
                loader = config.dataset.files[epochs_role].loader
                messages.append(ValidationMessage("error", f"Workflow {workflow.name!r} uses epochs role {epochs_role!r} with non-MNE loader {loader!r}."))
            label_column = workflow.label_column or _optional_str(config.dataset.defaults.get("label_column"))
            if label_column is None:
                messages.append(ValidationMessage("error", f"Workflow {workflow.name!r} is mne_time_decode but has no label_column."))
        elif not _referenced_roles(workflow):
            messages.append(ValidationMessage("warning", f"Workflow {workflow.name!r} does not reference any dataset file roles."))

    if check_files:
        for row in resolve_files(config):
            if row.required and not row.exists:
                messages.append(ValidationMessage("error", f"Required file for subject {row.subject!r}, role {row.role!r} does not exist: {row.path}"))
    return ConfigValidation(tuple(messages))


def validation_report_frame(validation: ConfigValidation) -> pd.DataFrame:
    """Return validation messages as a small tabular report."""

    return pd.DataFrame({"severity": message.severity, "message": message.message} for message in validation.messages)


def _select_workflow(config: DatasetConfig, workflow_name: str | None) -> WorkflowSpec:
    if workflow_name is None:
        if not config.workflows:
            raise DatasetConfigError("No workflow is defined.")
        return config.workflows[0]
    for workflow in config.workflows:
        if workflow.name == workflow_name:
            return workflow
    names = ", ".join(workflow.name for workflow in config.workflows) or "<none>"
    raise DatasetConfigError(f"Unknown workflow {workflow_name!r}; available workflows: {names}.")


def _relative_or_string(path: Path, base: Path | None) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _clean_options(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and value != ""}


def _metadata_options(config: DatasetConfig, workflow: WorkflowSpec) -> dict[str, Any]:
    merged: dict[str, Any] = dict(config.dataset.defaults)
    merged.update(workflow.options)
    explicit = {
        "label_column": workflow.label_column,
        "group_column": workflow.group_column,
        "source_column": workflow.source_column,
        "positive_pattern": workflow.positive_pattern,
        "negative_pattern": workflow.negative_pattern,
        "positive_label": workflow.positive_label,
        "negative_label": workflow.negative_label,
    }
    merged.update(_clean_options(explicit))
    return {key: value for key, value in merged.items() if key in _BENCHMARK_OPTION_COLUMNS or key in _METADATA_COLUMNS}


def to_benchmark_manifest_frame(
    config: DatasetConfig,
    *,
    workflow_name: str | None = None,
    relative_to: Path | str | None = None,
) -> pd.DataFrame:
    """Compile an ``mne_time_decode`` workflow to the existing benchmark CSV schema."""

    workflow = _select_workflow(config, workflow_name)
    if workflow.kind != "mne_time_decode":
        raise DatasetConfigError(f"Workflow {workflow.name!r} has kind {workflow.kind!r}; only mne_time_decode can be compiled to a benchmark manifest.")

    roles = config.dataset.files
    epochs_role = workflow.epochs or ("epochs" if "epochs" in roles else workflow.train)
    if epochs_role is None or epochs_role not in roles:
        raise DatasetConfigError(f"Workflow {workflow.name!r} must reference an epochs file role.")
    epochs_spec = roles[epochs_role]
    if epochs_spec.loader not in _MNE_EPOCHS_LOADERS:
        raise DatasetConfigError(f"File role {epochs_role!r} uses loader {epochs_spec.loader!r}; expected one of {sorted(_MNE_EPOCHS_LOADERS)}.")

    if workflow.metadata is not None:
        if workflow.metadata not in roles:
            raise DatasetConfigError(f"Workflow {workflow.name!r} references unknown metadata role {workflow.metadata!r}.")
        metadata_role = workflow.metadata
    else:
        metadata_role = "metadata" if "metadata" in roles else None

    if workflow.events is not None:
        if workflow.events not in roles:
            raise DatasetConfigError(f"Workflow {workflow.name!r} references unknown events role {workflow.events!r}.")
        events_role = workflow.events
    else:
        events_role = "events" if "events" in roles and metadata_role is None else None

    base = Path(relative_to) if relative_to is not None else None
    common_options = _metadata_options(config, workflow)

    rows: list[dict[str, Any]] = []
    for participant in config.dataset.participants:
        subject = _subject_for_participant(config, participant)
        row: dict[str, Any] = {
            "subject": subject,
            "participant": participant,
            "epochs": _relative_or_string(resolve_file_path(config, epochs_spec, participant=participant, subject=subject, workflow_name=workflow.name), base),
        }
        if metadata_role is not None:
            metadata_spec = roles[metadata_role]
            row["metadata_csv"] = _relative_or_string(resolve_file_path(config, metadata_spec, participant=participant, subject=subject, workflow_name=workflow.name), base)
        if events_role is not None:
            events_spec = roles[events_role]
            row["events_csv"] = _relative_or_string(resolve_file_path(config, events_spec, participant=participant, subject=subject, workflow_name=workflow.name), base)
        row.update(common_options)
        rows.append(row)
    return pd.DataFrame(rows)


def write_benchmark_manifest_csv(
    config: DatasetConfig,
    out_path: Path | str,
    *,
    workflow_name: str | None = None,
) -> pd.DataFrame:
    """Write an existing NeuRepTrace benchmark CSV generated from YAML/JSON config."""

    path = Path(out_path)
    frame = to_benchmark_manifest_frame(config, workflow_name=workflow_name, relative_to=path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a YAML/JSON dataset config and optionally emit a benchmark CSV."""

    parser = argparse.ArgumentParser(description="Validate a NeuRepTrace YAML/JSON dataset configuration.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-files", action="store_true", help="Require all configured files marked required=true to exist.")
    parser.add_argument("--workflow", help="Workflow name to use when writing a benchmark CSV.")
    parser.add_argument("--report-out", type=Path, help="Optional CSV file for validation messages.")
    parser.add_argument("--write-benchmark-manifest", type=Path, help="Write a benchmark CSV for an mne_time_decode workflow.")
    args = parser.parse_args(argv)

    config = load_dataset_config(args.config)
    validation = validate_dataset_config(config, check_files=args.check_files)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        validation_report_frame(validation).to_csv(args.report_out, index=False)

    for message in validation.messages:
        print(f"{message.severity}\t{message.message}")
    if not validation.messages:
        print("ok\tconfiguration is valid")
    if not validation.ok:
        return 1

    resolved = resolve_files(config, workflow_name=args.workflow)
    print(f"ok\tresolved {len(resolved)} file role(s) for {len(config.dataset.participants)} participant(s)")

    if args.write_benchmark_manifest is not None:
        frame = write_benchmark_manifest_csv(config, args.write_benchmark_manifest, workflow_name=args.workflow)
        print(f"ok\twrote benchmark manifest with {len(frame)} row(s): {args.write_benchmark_manifest}")
    return 0


__all__ = [
    "ConfigValidation",
    "DatasetConfig",
    "DatasetConfigError",
    "DatasetSpec",
    "FileSpec",
    "ResolvedFile",
    "ValidationMessage",
    "WorkflowSpec",
    "dataset_config_from_mapping",
    "expand_participants",
    "load_dataset_config",
    "resolve_file_path",
    "resolve_files",
    "to_benchmark_manifest_frame",
    "validate_dataset_config",
    "validation_report_frame",
    "write_benchmark_manifest_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())
