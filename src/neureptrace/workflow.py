"""Command-line helpers for declarative NeuRepTrace workflow specs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from neureptrace.workflow_spec import (
    WorkflowSpecError,
    check_workflow_files,
    load_workflow_spec,
    workflow_json_schema,
    workflow_spec_to_dict,
)


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path, help="Workflow specification in JSON, YAML, or YML format.")
    parser.add_argument("--check-files", action="store_true", help="Also check declared data/metadata paths without importing loaders.")
    parser.add_argument("--normalized-out", type=Path, help="Write the normalized workflow spec as JSON.")


def _write_normalized(path: Path, spec_dict: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec_dict, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_from_args(args: argparse.Namespace) -> int:
    try:
        spec = load_workflow_spec(args.config)
        messages = check_workflow_files(spec) if args.check_files else []
    except WorkflowSpecError as exc:
        print(f"error\t{args.config}: {exc}", file=sys.stderr)
        return 1

    if messages:
        for message in messages:
            print(f"error\t{message}", file=sys.stderr)
        return 1

    if args.normalized_out is not None:
        _write_normalized(args.normalized_out, workflow_spec_to_dict(spec))
        print(f"Wrote {args.normalized_out}")
    print(f"ok\t{args.config}\tworkflow={spec.workflow.kind}\tdataset={spec.dataset.id}")
    return 0


def validate_main(argv: Sequence[str] | None = None) -> int:
    """Validate a workflow configuration from the command line."""
    parser = argparse.ArgumentParser(description="Validate a NeuRepTrace workflow specification.")
    _add_validate_args(parser)
    return _validate_from_args(parser.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect workflow specifications.

    This intentionally stops short of executing workflows.  It establishes the
    shared upstream schema that dataset packages can target before full workflow
    runners are added.
    """
    parser = argparse.ArgumentParser(description="NeuRepTrace workflow specification utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a workflow JSON/YAML specification.")
    _add_validate_args(validate_parser)

    show_parser = subparsers.add_parser("show", help="Print a normalized workflow specification as JSON.")
    show_parser.add_argument("config", type=Path, help="Workflow specification in JSON, YAML, or YML format.")

    subparsers.add_parser("schema", help="Print the JSON Schema for workflow specifications.")

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate_from_args(args)
    if args.command == "show":
        try:
            spec = load_workflow_spec(args.config)
        except WorkflowSpecError as exc:
            print(f"error\t{args.config}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(workflow_spec_to_dict(spec), indent=2, sort_keys=True))
        return 0
    if args.command == "schema":
        print(json.dumps(workflow_json_schema(), indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
