"""Command-line helpers for NeuRepTrace dataset specifications."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from neureptrace.dataset_spec import expand_manifest, load_dataset_spec, parse_subjects, validate_dataset_spec
from neureptrace.datasets.pymegdec import (
    add_pymegdec_bushmeg_dataset_spec_arguments,
    write_pymegdec_bushmeg_dataset_spec_from_args,
)


def _parse_subject_arg(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return parse_subjects(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and expand NeuRepTrace YAML/JSON dataset specifications.")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="Validate a dataset spec and print or write an inventory table.")
    validate.add_argument("spec", help="Dataset YAML/JSON specification.")
    validate.add_argument("--subjects", default=None, help="Optional subject subset, e.g. 1-4,6,8.")
    validate.add_argument("--split", action="append", dest="splits", default=None, help="Split to validate. Repeat to validate multiple splits.")
    validate.add_argument("--root", default=None, help="Override the data root resolved by the spec.")
    validate.add_argument("--require-files", action="store_true", help="Fail if any resolved data or metadata file is missing.")
    validate.add_argument("--out", default=None, help="Optional CSV path for the validation inventory.")

    manifest = subparsers.add_parser("manifest", help="Expand a dataset spec into a benchmark-style manifest CSV.")
    manifest.add_argument("spec", help="Dataset YAML/JSON specification.")
    manifest.add_argument("--workflow", default=None, help="Workflow key whose manifest defaults should be merged.")
    manifest.add_argument("--subjects", default=None, help="Optional subject subset, e.g. 1-4,6,8.")
    manifest.add_argument("--split", default=None, help="Single split to expand. Overrides workflow.split.")
    manifest.add_argument("--root", default=None, help="Override the data root resolved by the spec.")
    manifest.add_argument("--out", required=True, help="Output CSV manifest path.")

    pymegdec_bushmeg = subparsers.add_parser("pymegdec-bushmeg", help="Write the canonical PyMEGDec/BUSH-MEG dataset spec.")
    add_pymegdec_bushmeg_dataset_spec_arguments(pymegdec_bushmeg)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset-spec command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "pymegdec-bushmeg":
        return write_pymegdec_bushmeg_dataset_spec_from_args(args)

    spec = load_dataset_spec(args.spec)
    if args.command == "validate":
        inventory = validate_dataset_spec(
            spec,
            subjects=_parse_subject_arg(args.subjects),
            splits=args.splits,
            require_files=args.require_files,
            root=args.root,
        )
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            inventory.to_csv(out, index=False)
            print(f"Wrote dataset inventory to {out}")
        else:
            print(inventory.to_string(index=False))
        return 0

    if args.command == "manifest":
        manifest = expand_manifest(spec, workflow=args.workflow, subjects=_parse_subject_arg(args.subjects), split=args.split, root=args.root)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out, index=False)
        print(f"Wrote dataset manifest with {len(manifest)} rows to {out}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
