from __future__ import annotations

import json

import pytest

from neureptrace.cli import main


def test_list_format_requires_list_commands() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--list-format", "json"])

    assert exc_info.value.code == 2


def test_json_list_format_still_works_with_list_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-commands", "--list-format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "commands" in payload
    assert any(command["command"] == "benchmark" for command in payload["commands"])
