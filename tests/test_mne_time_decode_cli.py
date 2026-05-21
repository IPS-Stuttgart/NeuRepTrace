import sys

import pytest

from neureptrace import cli, mne_time_decode_cli, mne_time_decode_foldlocal_cli


@pytest.mark.parametrize(
    "module",
    [mne_time_decode_cli, mne_time_decode_foldlocal_cli],
)
def test_mne_time_decode_direct_help_does_not_conflict(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["neureptrace-mne-time-decode", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 0
    assert "--fieldtrip-root-path" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    ["mne-time-decode", "mne-time-decode-base"],
)
def test_grouped_mne_time_decode_help_does_not_conflict(command, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([command, "--help"])

    assert exc_info.value.code == 0
    assert "--fieldtrip-root-path" in capsys.readouterr().out
