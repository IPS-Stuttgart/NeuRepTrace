"""YAML/JSON dataset specifications for dataset-specific NeuRepTrace adapters."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_SCHEMA_VERSION = 1
_RANGE_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_ENV_PATTERN = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class DatasetFile:
    """Resolved file for one participant and one logical dataset role."""

    participant: str
    role: str
    file_role: str
    path: Path
    exists: bool


@dataclass(frozen=True)
class DatasetValidation:
    """Validation result for one dataset-spec scope."""

    scope: str
    ok: bool
    messages: list[str]


def load_dataset_spec(spec_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a NeuRepTrace dataset spec from YAML or JSON."""

    path = Path(spec_path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency is declared by the package.
            raise RuntimeError("YAML dataset specs require PyYAML. Install NeuRepTrace with its runtime dependencies.") from exc
        loaded = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported dataset spec extension '{path.suffix}'. Use .yml, .yaml, or .json.")

    if not isinstance(loaded, dict):
        raise ValueError("Dataset spec must load to a mapping/object at the top level.")
    return dict(loaded)


def expand_participant_ids(ids: Iterable[Any]) -> tuple[str, ...]:
    """Expand participant identifiers, including compact range strings like ``1-4``."""

    expanded: list[str] = []
    for token in ids:
        expanded.extend(_expand_participant_token(token))

    deduplicated: list[str] = []
    seen: set[str] = set()
    for participant in expanded:
        if participant in seen:
            continue
        deduplicated.append(participant)
        seen.add(participant)
    return tuple(deduplicated)


def resolve_dataset_files(spec: Mapping[str, Any], *, spec_dir: str | os.PathLike[str] | None = None) -> list[DatasetFile]:
    """Resolve participant file templates into concrete paths."""

    base_dir = Path(spec_dir) if spec_dir is not None else Path.cwd()
    root = _dataset_root(spec, base_dir=base_dir)
    participants = _participant_ids(spec)
    files = _file_templates(spec)
    roles = _role_to_file_roles(spec, files)

    resolved: list[DatasetFile] = []
    for participant in participants:
        format_values = _format_values_for_participant(participant)
        for role, file_role in roles.items():
            template = files[file_role]
            relative = _format_template(template, format_values)
            path = _resolve_path(relative, base_dir=root)
            resolved.append(
                DatasetFile(
                    participant=participant,
                    role=role,
                    file_role=file_role,
                    path=path,
                    exists=path.exists(),
                )
            )
    return resolved


def build_dataset_file_table(spec_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Return the resolved dataset file table for a YAML/JSON spec."""

    path = Path(spec_path)
    spec = load_dataset_spec(path)
    rows = [
        {
            "participant": dataset_file.participant,
            "role": dataset_file.role,
            "file_role": dataset_file.file_role,
            "path": str(dataset_file.path),
            "exists": dataset_file.exists,
        }
        for dataset_file in resolve_dataset_files(spec, spec_dir=path.parent)
    ]
    return pd.DataFrame(rows, columns=["participant", "role", "file_role", "path", "exists"])


def validate_dataset_spec(
    spec_path: str | os.PathLike[str],
    *,
    check_exists: bool = True,
) -> list[DatasetValidation]:
    """Validate a dataset spec and, by default, the files it resolves to."""

    path = Path(spec_path)
    spec = load_dataset_spec(path)
    return validate_loaded_dataset_spec(spec, spec_dir=path.parent, check_exists=check_exists)


def validate_loaded_dataset_spec(
    spec: Mapping[str, Any],
    *,
    spec_dir: str | os.PathLike[str] | None = None,
    check_exists: bool = True,
) -> list[DatasetValidation]:
    """Validate a loaded dataset spec mapping."""

    base_dir = Path(spec_dir) if spec_dir is not None else Path.cwd()
    validations: list[DatasetValidation] = []

    schema_messages = _validate_schema_version(spec)
    validations.append(DatasetValidation(scope="schema", ok=not schema_messages, messages=schema_messages))

    dataset_messages = _validate_dataset_section(spec, base_dir=base_dir, check_exists=check_exists)
    validations.append(DatasetValidation(scope="dataset", ok=not dataset_messages, messages=dataset_messages))

    participant_messages = _validate_participant_section(spec)
    validations.append(DatasetValidation(scope="participants", ok=not participant_messages, messages=participant_messages))

    role_messages = _validate_role_section(spec)
    validations.append(DatasetValidation(scope="roles", ok=not role_messages, messages=role_messages))

    if any(validation.messages for validation in validations):
        return validations

    for dataset_file in resolve_dataset_files(spec, spec_dir=base_dir):
        messages: list[str] = []
        if check_exists and not dataset_file.exists:
            messages.append(f"file does not exist: {dataset_file.path}")
        scope = f"file:{dataset_file.participant}:{dataset_file.role}"
        validations.append(DatasetValidation(scope=scope, ok=not messages, messages=messages))

    return validations


def validation_report_frame(validations: Sequence[DatasetValidation]) -> pd.DataFrame:
    """Return a tabular dataset-spec validation report."""

    rows = [
        {
            "scope": validation.scope,
            "ok": validation.ok,
            "messages": " | ".join(validation.messages),
        }
        for validation in validations
    ]
    return pd.DataFrame(rows, columns=["scope", "ok", "messages"])


def _validate_schema_version(spec: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    version = spec.get("schema_version")
    if version is None:
        messages.append("schema_version is missing")
        return messages

    parsed_version = _parse_schema_version(version)
    if parsed_version is None:
        messages.append(f"schema_version must be an integer value; got {version!r}")
    elif parsed_version != SUPPORTED_SCHEMA_VERSION:
        messages.append(f"unsupported schema_version={version}; expected {SUPPORTED_SCHEMA_VERSION}")
    return messages


def _parse_schema_version(version: Any) -> int | None:
    if isinstance(version, bool):
        return None
    if isinstance(version, Integral):
        return int(version)
    if isinstance(version, Real) and float(version).is_integer():
        return int(version)
    if isinstance(version, str):
        text = version.strip()
        if text.isdigit():
            return int(text)
    return None


def _validate_dataset_section(spec: Mapping[str, Any], *, base_dir: Path, check_exists: bool) -> list[str]:
    messages: list[str] = []
    dataset = spec.get("dataset")
    if not isinstance(dataset, Mapping):
        return ["dataset section must be a mapping"]

    if _missing(dataset.get("id")):
        messages.append("dataset.id is missing")
    if _missing(dataset.get("format")):
        messages.append("dataset.format is missing")

    root_value = dataset.get("root")
    if _missing(root_value):
        messages.append("dataset.root is missing")
    else:
        root_text = str(root_value)
        expanded_root_text = os.path.expandvars(os.path.expanduser(root_text))
        if _has_unexpanded_env(expanded_root_text):
            messages.append(f"dataset.root contains unresolved environment variable(s): {root_text}")
        elif check_exists:
            root = _resolve_path(expanded_root_text, base_dir=base_dir)
            if not root.exists():
                messages.append(f"dataset.root does not exist: {root}")
            elif not root.is_dir():
                messages.append(f"dataset.root is not a directory: {root}")
    return messages


def _validate_participant_section(spec: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    participants = spec.get("participants")
    if not isinstance(participants, Mapping):
        return ["participants section must be a mapping"]

    ids = participants.get("ids")
    if not isinstance(ids, Iterable) or isinstance(ids, (str, bytes)):
        messages.append("participants.ids must be a list")
    else:
        try:
            participant_ids = expand_participant_ids(ids)
        except ValueError as exc:
            messages.append(str(exc))
        else:
            if not participant_ids:
                messages.append("participants.ids must not be empty")

    files = participants.get("files")
    if not isinstance(files, Mapping) or not files:
        messages.append("participants.files must be a non-empty mapping")
    else:
        for file_role, template in files.items():
            if _missing(file_role):
                messages.append("participants.files contains an empty file role")
            if _missing(template):
                messages.append(f"participants.files.{file_role} is empty")
            elif not _template_mentions_participant(str(template)):
                messages.append(f"participants.files.{file_role} should include a participant placeholder")
    return messages


def _validate_role_section(spec: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    roles = spec.get("roles")
    if roles is None:
        return messages
    if not isinstance(roles, Mapping):
        return ["roles section must be a mapping when present"]

    files = _file_templates(spec, allow_missing=True)
    for role, value in roles.items():
        if _missing(role):
            messages.append("roles contains an empty role name")
        if isinstance(value, Mapping):
            file_role = value.get("file_role", role)
        else:
            file_role = value
        if _missing(file_role):
            messages.append(f"roles.{role}.file_role is missing")
        elif files and str(file_role) not in files:
            messages.append(f"roles.{role}.file_role='{file_role}' is not defined in participants.files")
    return messages


def _participant_ids(spec: Mapping[str, Any]) -> tuple[str, ...]:
    participants = _mapping(spec, "participants")
    ids = participants.get("ids")
    if not isinstance(ids, Iterable) or isinstance(ids, (str, bytes)):
        raise ValueError("participants.ids must be a list")
    return expand_participant_ids(ids)


def _file_templates(spec: Mapping[str, Any], *, allow_missing: bool = False) -> dict[str, str]:
    participants = _mapping(spec, "participants")
    files = participants.get("files")
    if not isinstance(files, Mapping):
        if allow_missing:
            return {}
        raise ValueError("participants.files must be a mapping")
    return {str(key): str(value) for key, value in files.items()}


def _role_to_file_roles(spec: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, str]:
    roles = spec.get("roles")
    if roles is None:
        return {file_role: file_role for file_role in files}
    if not isinstance(roles, Mapping):
        raise ValueError("roles must be a mapping")

    role_to_file_role: dict[str, str] = {}
    for role, value in roles.items():
        if isinstance(value, Mapping):
            file_role = value.get("file_role", role)
        else:
            file_role = value
        role_to_file_role[str(role)] = str(file_role)
    return role_to_file_role


def _dataset_root(spec: Mapping[str, Any], *, base_dir: Path) -> Path:
    dataset = _mapping(spec, "dataset")
    root = dataset.get("root")
    if _missing(root):
        raise ValueError("dataset.root is missing")
    return _resolve_path(os.path.expandvars(os.path.expanduser(str(root))), base_dir=base_dir)


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} section must be a mapping")
    return value


def _missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _has_unexpanded_env(value: str) -> bool:
    return bool(_ENV_PATTERN.search(value))


def _resolve_path(value: str | os.PathLike[str], *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _expand_participant_token(token: Any) -> list[str]:
    if isinstance(token, Mapping):
        if "range" in token:
            return _expand_range_value(token["range"])
        if "id" in token:
            return [str(token["id"])]
        raise ValueError(f"Unsupported participant token mapping: {token}")

    if isinstance(token, int):
        return [str(token)]
    if isinstance(token, float) and token.is_integer():
        return [str(int(token))]

    text = str(token).strip()
    if not text:
        raise ValueError("participants.ids contains an empty identifier")
    return _expand_range_value(text)


def _expand_range_value(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Participant range sequences must contain exactly two values: {value}")
        return _range_ids(str(value[0]), str(value[1]))

    text = str(value).strip()
    match = _RANGE_PATTERN.match(text)
    if match is None:
        return [text]
    return _range_ids(match.group(1), match.group(2))


def _range_ids(start_text: str, stop_text: str) -> list[str]:
    start = int(start_text)
    stop = int(stop_text)
    step = 1 if start <= stop else -1
    width = max(len(start_text), len(stop_text)) if start_text.startswith("0") or stop_text.startswith("0") else 0
    return [_format_participant_number(value, width) for value in range(start, stop + step, step)]


def _format_participant_number(value: int, width: int) -> str:
    return str(value).zfill(width) if width else str(value)


def _format_values_for_participant(participant: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "participant": participant,
        "participant_id": participant,
    }
    if participant.isdigit():
        values["participant_number"] = int(participant)
    return values


def _format_template(template: str, values: Mapping[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        available = ", ".join(sorted(values))
        raise ValueError(f"Unknown placeholder in file template '{template}': {exc}. Available placeholders: {available}") from exc


def _template_mentions_participant(template: str) -> bool:
    return any(
        placeholder in template
        for placeholder in (
            "{participant}",
            "{participant:",
            "{participant_id}",
            "{participant_id:",
            "{participant_number}",
            "{participant_number:",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line interface for dataset spec validation and file resolution."""

    parser = argparse.ArgumentParser(description="Validate and inspect NeuRepTrace YAML/JSON dataset specifications.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a dataset spec and resolved files.")
    validate_parser.add_argument("spec", type=Path)
    validate_parser.add_argument("--no-check-exists", action="store_true", help="Validate schema and path templates without checking that files exist.")
    validate_parser.add_argument("--report-out", type=Path, help="Optional CSV validation report path.")

    list_parser = subparsers.add_parser("list-files", help="Print the resolved participant file table.")
    list_parser.add_argument("spec", type=Path)
    list_parser.add_argument("--format", choices=("csv", "json"), default="csv")

    args = parser.parse_args(argv)

    if args.command == "validate":
        validations = validate_dataset_spec(args.spec, check_exists=not args.no_check_exists)
        report = validation_report_frame(validations)
        if args.report_out is not None:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(args.report_out, index=False)
            print(f"Wrote {args.report_out}")
        for row in report.itertuples(index=False):
            status = "ok" if row.ok else "error"
            detail = "" if row.ok else f": {row.messages}"
            print(f"{status}\t{row.scope}{detail}")
        return 0 if all(validation.ok for validation in validations) else 1

    if args.command == "list-files":
        frame = build_dataset_file_table(args.spec)
        if args.format == "json":
            print(frame.to_json(orient="records", indent=2))
        else:
            frame.to_csv(sys.stdout, index=False)
        return 0

    raise RuntimeError(f"Unhandled dataset spec command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
