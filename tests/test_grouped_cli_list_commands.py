from __future__ import annotations

import json

import pytest

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


def test_grouped_cli_json_list_exposes_aliases(capsys) -> None:
    assert cli.main(["--list-commands", "--list-format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    commands = {record["command"]: record for record in payload["commands"]}

    assert commands["doctor"] == {
        "command": "doctor",
        "module": "neureptrace.doctor",
        "aliases": ["check", "env"],
    }
    assert commands["check"] == {
        "command": "check",
        "module": "neureptrace.doctor",
        "aliases": ["doctor", "env"],
    }
    assert commands["stimulus-detect"]["module"] == "neureptrace.stimulus_detection"
    assert "stimulus-detection" in commands["stimulus-detect"]["aliases"]


def test_grouped_cli_list_rejects_extra_workflow_args() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--list-commands", "benchmark"])

    assert exc_info.value.code == 2


def test_command_listing_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported command listing format: yaml"):
        cli._command_listing("yaml")
