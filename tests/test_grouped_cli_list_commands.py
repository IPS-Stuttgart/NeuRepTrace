from __future__ import annotations

from neureptrace import cli


def test_grouped_cli_lists_commands(capsys) -> None:
    assert cli.main(["--list-commands"]) == 0

    output = capsys.readouterr().out
    assert "Available commands:" in output
    assert "  benchmark" in output
    assert "neureptrace.benchmark" in output
    assert "  temporal-smoothing" in output
    assert "neureptrace.temporal_smoothing" in output


def test_grouped_cli_list_alias_lists_commands(capsys) -> None:
    assert cli.main(["--list"]) == 0

    output = capsys.readouterr().out
    assert "Available commands:" in output
    assert "  doctor" in output
    assert "neureptrace.doctor" in output
