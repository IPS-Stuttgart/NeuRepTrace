from __future__ import annotations

import json

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


def test_command_listing_rejects_unknown_format() -> None:
    try:
        cli._command_listing("yaml")
    except ValueError as exc:
        assert "Unsupported command listing format: yaml" in str(exc)
    else:  # pragma: no cover - defensive assertion helper
        raise AssertionError("_command_listing() accepted an unsupported format")
