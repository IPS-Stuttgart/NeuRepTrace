"""Grouped command-line interface for NeuRepTrace workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import import_module

COMMAND_MODULES = {
    "benchmark": "neureptrace.benchmark",
    "bushmeg-data": "neureptrace.bushmeg_data",
    "bushmeg-source-loso": "neureptrace.bushmeg_source_loso",
    "bushmeg-loso-decode": "neureptrace.bushmeg_loso_decode",
    "continuous-stimulus-scan": "neureptrace.continuous_stimulus_scan",
    "dataset": "neureptrace.dataset_spec_cli",
    "dataset-config": "neureptrace.dataset_config",
    "dataset-spec": "neureptrace.datasets.spec",
    "decode-from-config": "neureptrace.decode_from_config",
    "dataset-manifest": "neureptrace.dataset_manifest",
    "epoch-transfer-decode": "neureptrace.epoch_transfer_decode",
    "ensemble-observations": "neureptrace.observation_ensemble",
    "event-detect": "neureptrace.event_detection",
    "event-detection": "neureptrace.event_detection",
    "fieldtrip-to-mne": "neureptrace.fieldtrip_mat",
    "loso-time-decode": "neureptrace.loso_time_decode",
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
    "plot-time-decode": "neureptrace.plot_time_decode",
    "results": "neureptrace.results",
    "stimulus-detect": "neureptrace.stimulus_detection",
    "stimulus-detection": "neureptrace.stimulus_detection",
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


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch installed NeuRepTrace subcommands."""
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in COMMAND_MODULES:
        return _run_module_main(argv[0], argv[1:])

    parser = argparse.ArgumentParser(description="NeuRepTrace command-line interface.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMAND_MODULES),
        help="Workflow to run. Pass '<command> --help' for command-specific options.",
    )
    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return _run_module_main(args.command, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
