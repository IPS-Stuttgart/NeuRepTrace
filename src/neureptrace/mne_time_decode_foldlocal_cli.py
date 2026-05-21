"""Console entry point for fold-local MNE time decoding."""

from __future__ import annotations

from neureptrace import mne_time_decode as _base
from neureptrace import mne_time_decode_foldlocal as _foldlocal
from neureptrace._mne_cli_argparse import run_with_conflict_resolving_parser


def _main_with_foldlocal_decoder() -> None:
    original_run_time_resolved_decode = _base.run_time_resolved_decode
    _base.run_time_resolved_decode = _foldlocal.run_time_resolved_decode
    try:
        return _base.main()
    finally:
        _base.run_time_resolved_decode = original_run_time_resolved_decode


def main() -> None:
    """Run the fold-local MNE time-decode CLI."""

    return run_with_conflict_resolving_parser(_base, _main_with_foldlocal_decoder)
