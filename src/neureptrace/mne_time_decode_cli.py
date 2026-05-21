"""Console entry point for the base MNE time-decoding workflow."""

from __future__ import annotations

from neureptrace import mne_time_decode as _base
from neureptrace._mne_cli_argparse import run_with_conflict_resolving_parser


def main() -> None:
    """Run the base MNE time-decode CLI."""

    return run_with_conflict_resolving_parser(_base, _base.main)
