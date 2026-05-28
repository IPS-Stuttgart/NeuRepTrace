"""Grouped command-line interface for NeuRepTrace workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from difflib import get_close_matches
from importlib import import_module

from neureptrace import __version__

COMMAND_MODULES = {
    "benchmark": "neureptrace.benchmark",
    "bushmeg-artifact-diff": "neureptrace.bushmeg_artifact_diff",
    "bushmeg-data": "neureptrace.bushmeg_data",
    "bushmeg-covariance-loso": "neureptrace.bushmeg_covariance_loso",
    "bushmeg-source-loso": "neureptrace.bushmeg_source_loso",
    "bushmeg-source-loso-ensemble": "neureptrace.bushmeg_source_loso_ensemble",
    "bushmeg-supervised-lowrank-loso": "neureptrace.bushmeg_supervised_lowrank_loso",
    "bushmeg-diagnostics": "neureptrace.bushmeg_diagnostics",
    "bushmeg-loso-decode": "neureptrace.bushmeg_loso_decode",
    "check": "neureptrace.doctor",
    "continuous-stimulus-scan": "neureptrace.continuous_stimulus_scan",
    "dataset": "neureptrace.dataset_spec_cli",
    "dataset-config": "neureptrace.dataset_config",
    "dataset-spec": "neureptrace.datasets.spec",
    "decode-from-config": "neureptrace.decode_from_config",
    "dataset-manifest": "neureptrace.dataset_manifest",
    "doctor": "neureptrace.doctor",
    "env": "neureptrace.doctor",
    "epoch-transfer-decode": "neureptrace.epoch_transfer_decode",
    "emission-compare": "neureptrace.emission_compare",
    "ensemble-observations": "neureptrace.observation_ensemble",
    "event-detect": "neureptrace.event_detection",
    "event-detection": "neureptrace.event_detection",
    "fieldtrip-to-mne": "neureptrace.fieldtrip_mat",
    "synthetic-fieldtrip": "neureptrace.synthetic_fieldtrip",
    "loso-time-decode": "neureptrace.loso_time_decode",
    "loso-observation-diagnostics": "neureptrace.loso_observation_diagnostics",
    "metadata": "neureptrace.metadata",
    "mne-transfer-decode": "neureptrace.epoch_transfer_decode",
    "mne-time-decode": "neureptrace.mne_time_decode_foldlocal_cli",
    "mne-time-decode-base": "neureptrace.mne_time_decode_cli",
    "mne-time-decode-ensemble": "neureptrace.mne_time_decode_ensemble",
    "observation-ensemble": "neureptrace.observation_ensemble",
    "observation-schema": "neureptrace.observation_schema",
    "onset-detect": "neureptrace.onset_detection",
    "onset-detection": "neureptrace.onset_detection",
    "openneuro-meg": "neureptrace.openneuro_meg",
    "openneuro-diagnostics": "neureptrace.openneuro_decode_diagnostics",
    "openneuro-resilient": "neureptrace.openneuro_resilient",
    "plot-time-decode": "neureptrace.plot_time_decode",
    "pymegdec-bushmeg-spec": "neureptrace.datasets.pymegdec",
    "probability-stacking": "neureptrace.probability_stacking",
    "results": "neureptrace.results",
    "source-oof-stacking": "neureptrace.probability_stacking",
    "stimulus-detect": "neureptrace.stimulus_detection",
    "stimulus-detection": "neureptrace.stimulus_detection",
    "temporal-decision-decode": "neureptrace.temporal_decision_decode",
    "temporal-model": "neureptrace.temporal_model",
    "temporal-smoothing": "neureptrace.temporal_smoothing",
    "temporal-state-workflow": "neureptrace.temporal_state_workflow",
    "transfer-decode": "neureptrace.epoch_transfer_decode",
    "transfer-from-config": "neureptrace.transfer_from_config",
    "time-transfer-decode": "neureptrace.time_transfer_decode",
    "validate-dataset-config": "neureptrace.dataset_config",
    "validate-manifest": "neureptrace.validate_manifest",
    "validate-observations": "neureptrace.observation_schema",
}


def _aliases_by_module() -> dict[str, tuple[str, ...]]:
    """Return grouped command aliases for each backing module."""

    aliases: dict[str, list[str]] = {}
    for command, module_name in COMMAND_MODULES.items():
        aliases.setdefault(module_name, []).append(command)
    return {module_name: tuple(sorted(commands)) for module_name, commands in aliases.items()}


def _command_records() -> list[dict[str, object]]:
    """Return a machine-readable inventory of grouped commands."""

    aliases_by_module = _aliases_by_module()
    records: list[dict[str, object]] = []
    for command in sorted(COMMAND_MODULES):
        module_name = COMMAND_MODULES[command]
        records.append(
            {
                "command": command,
                "module": module_name,
                "aliases": [alias for alias in aliases_by_module[module_name] if alias != command],
            }
        )
    return records


def _command_listing(output_format: str = "text") -> str:
    """Return a stable inventory of grouped commands."""

    if output_format == "json":
        return json.dumps({"commands": _command_records()}, indent=2, sort_keys=True)
    if output_format != "text":
        raise ValueError(f"Unsupported command listing format: {output_format}")

    width = max((len(command) for command in COMMAND_MODULES), default=0)
    rows = [
        f"  {command:<{width}}  {COMMAND_MODULES[command]}"
        for command in sorted(COMMAND_MODULES)
    ]
    return "\n".join(["Available commands:", *rows])


def _format_unknown_command_error(command: str) -> str:
    """Return an actionable error message for a mistyped grouped command."""

    close_matches = get_close_matches(command, sorted(COMMAND_MODULES), n=3, cutoff=0.6)
    message = f"unknown command '{command}'"
    if len(close_matches) == 1:
        message += f". Did you mean '{close_matches[0]}'?"
    elif close_matches:
        suggestions = ", ".join(f"'{match}'" for match in close_matches)
        message += f". Did you mean one of: {suggestions}?"
    return f"{message} Use --list-commands to inspect available workflows."


def _run_module_main(command: str, argv: Sequence[str]) -> int:
    """Run a NeuRepTrace module-level ``main`` as a grouped subcommand."""
    module = import_module(COMMAND_MODULES[command])
    module_main = getattr(module, "main", None)
    if module_main is None:
        raise RuntimeError(f"Command '{command}' is backed by {module.__name__}, which has no main() function.")

    original_argv = sys.argv
    sys.argv = [f"neureptrace {command}", *argv]
    try:
        result = module_main()
    finally:
        sys.argv = original_argv

    return int(result) if isinstance(result, int) else 0


def _handle_help_command(argv: Sequence[str]) -> int:
    """Show grouped-command help without requiring users to remember argument order."""

    if not argv:
        print(_command_listing())
        return 0

    if len(argv) != 1:
        print("usage: neureptrace help [command]", file=sys.stderr)
        raise SystemExit(2)

    command = argv[0]
    if command in {"--list", "--list-commands"}:
        print(_command_listing())
        return 0

    if command in {"-h", "--help"}:
        print("usage: neureptrace help [command]\n")
        print("Show command-specific help for a grouped NeuRepTrace workflow.\n")
        print(_command_listing())
        return 0

    if command not in COMMAND_MODULES:
        print(_format_unknown_command_error(command), file=sys.stderr)
        raise SystemExit(2)

    return _run_module_main(command, ["--help"])


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch installed NeuRepTrace subcommands."""
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "help":
        return _handle_help_command(argv[1:])

    if argv and argv[0] in COMMAND_MODULES:
        return _run_module_main(argv[0], argv[1:])

    if argv and not argv[0].startswith("-"):
        print(_format_unknown_command_error(argv[0]), file=sys.stderr)
        raise SystemExit(2)

    parser = argparse.ArgumentParser(description="NeuRepTrace command-line interface.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"neureptrace {__version__}",
        help="Show the installed NeuRepTrace version and exit.",
    )
    parser.add_argument(
        "--list-commands",
        "--list",
        dest="list_commands",
        action="store_true",
        help="List grouped workflow commands and their backing modules, then exit.",
    )
    parser.add_argument(
        "--list-format",
        choices=("text", "json"),
        default="text",
        help="Output format for --list-commands. Use json for scripts that need command aliases and module targets.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        metavar="command",
        help="Workflow to run. Use --list-commands to inspect available workflows; pass '<command> --help' for command-specific options.",
    )
    args, remaining = parser.parse_known_args(argv)

    if remaining and args.command is None:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    if args.list_commands:
        print(_command_listing(args.list_format))
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command not in COMMAND_MODULES:
        parser.error(_format_unknown_command_error(args.command))

    return _run_module_main(args.command, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
