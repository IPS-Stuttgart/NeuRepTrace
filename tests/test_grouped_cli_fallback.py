from __future__ import annotations

import pytest

from neureptrace import cli


def test_grouped_cli_dispatches_command_from_argparse_fallback(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run_module_main(command: str, argv) -> int:
        calls.append((command, tuple(argv)))
        return 17

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)

    result = cli.main(["--list-format", "text", "stimulus-detect", "--score-mode", "class_probability"])

    assert result == 17
    assert calls == [("stimulus-detect", ("--score-mode", "class_probability"))]


def test_grouped_cli_rejects_list_format_without_list_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--list-format", "json", "stimulus-detect"])

    assert exc_info.value.code == 2
    assert "--list-format can only be used with --list-commands/--list" in capsys.readouterr().err
