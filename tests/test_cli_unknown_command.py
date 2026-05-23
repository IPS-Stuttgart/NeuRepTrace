from __future__ import annotations

from neureptrace import cli


def test_unknown_grouped_cli_command_suggests_close_match(capsys):
    assert cli.main(["mne-time-deocde", "--help"]) == 2

    captured = capsys.readouterr()
    assert "Unknown command 'mne-time-deocde'" in captured.err
    assert "mne-time-decode" in captured.err
    assert "--list-commands" in captured.err
    assert captured.out == ""


def test_unknown_grouped_cli_command_does_not_mask_options(capsys):
    assert cli.main(["--list-commands"]) == 0

    captured = capsys.readouterr()
    assert "Available commands:" in captured.out
    assert captured.err == ""
