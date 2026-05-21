"""Console entry point for fold-local MNE time decoding."""

from __future__ import annotations

from neureptrace import mne_time_decode as _base
from neureptrace import mne_time_decode_foldlocal as _foldlocal
from neureptrace._mne_cli_argparse import run_with_conflict_resolving_parser


def main() -> None:
    """Run the fold-local MNE time-decode CLI."""

    return run_with_conflict_resolving_parser(_base, _foldlocal.main)
