import argparse
import sys

import pytest

from neureptrace import mne_time_decode, mne_time_decode_foldlocal_cli


def test_foldlocal_cli_help_restores_global_state(monkeypatch):
    original_parser = mne_time_decode.argparse.ArgumentParser
    original_run = mne_time_decode.run_time_resolved_decode
    monkeypatch.setattr(sys, "argv", ["neureptrace-mne-time-decode", "--help"])

    with pytest.raises(SystemExit):
        mne_time_decode_foldlocal_cli.main()

    assert mne_time_decode.argparse.ArgumentParser is original_parser
    assert mne_time_decode.argparse.ArgumentParser is argparse.ArgumentParser
    assert mne_time_decode.run_time_resolved_decode is original_run
