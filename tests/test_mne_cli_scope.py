import argparse
import sys

import pytest

from neureptrace import mne_time_decode, mne_time_decode_foldlocal_cli
from neureptrace._mne_cli_argparse import run_with_conflict_resolving_parser


def test_foldlocal_cli_help_restores_global_state(monkeypatch):
    original_parser = mne_time_decode.argparse.ArgumentParser
    original_run = mne_time_decode.run_time_resolved_decode
    monkeypatch.setattr(sys, "argv", ["neureptrace-mne-time-decode", "--help"])

    with pytest.raises(SystemExit):
        mne_time_decode_foldlocal_cli.main()

    assert mne_time_decode.argparse.ArgumentParser is original_parser
    assert mne_time_decode.argparse.ArgumentParser is argparse.ArgumentParser
    assert mne_time_decode.run_time_resolved_decode is original_run


def test_conflict_resolving_parser_wrapper_is_reentrant():
    original_init = mne_time_decode.argparse.ArgumentParser.__init__
    outer_wrappers: list[object] = []
    wrapper_identities: list[bool] = []

    def inner_main() -> int:
        wrapper_identities.append(mne_time_decode.argparse.ArgumentParser.__init__ is outer_wrappers[0])
        parser = mne_time_decode.argparse.ArgumentParser()
        assert parser.conflict_handler == "resolve"
        return 11

    def outer_main() -> int:
        outer_wrapper = mne_time_decode.argparse.ArgumentParser.__init__
        outer_wrappers.append(outer_wrapper)
        assert outer_wrapper is not original_init
        return run_with_conflict_resolving_parser(mne_time_decode, inner_main)

    assert run_with_conflict_resolving_parser(mne_time_decode, outer_main) == 11
    assert wrapper_identities == [True]
    assert mne_time_decode.argparse.ArgumentParser.__init__ is original_init
