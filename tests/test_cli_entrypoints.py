from __future__ import annotations

import json
import tomllib
from pathlib import Path

from neureptrace.cli import COMMAND_MODULES, _command_listing

REPO_ROOT = Path(__file__).resolve().parents[1]


def _poetry_scripts() -> dict[str, str]:
    """Return Poetry console-script declarations from the project metadata."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return dict(pyproject["tool"]["poetry"]["scripts"])


def _workflow_script_commands() -> dict[str, str]:
    """Return direct NeuRepTrace workflow scripts keyed by grouped command name."""

    return {
        script_name.removeprefix("neureptrace-"): target
        for script_name, target in _poetry_scripts().items()
        if script_name.startswith("neureptrace-")
    }


def test_direct_workflow_scripts_are_available_in_grouped_cli() -> None:
    """Keep direct console scripts discoverable through the grouped CLI."""

    script_commands = _workflow_script_commands()
    missing = sorted(set(script_commands) - set(COMMAND_MODULES))

    assert not missing, f"Add grouped CLI aliases for console scripts: {', '.join(missing)}"


def test_grouped_cli_targets_match_direct_console_scripts() -> None:
    """Prevent direct and grouped commands from dispatching to different modules."""

    mismatched = {
        command: {"script_target": target.split(":", 1)[0], "grouped_target": COMMAND_MODULES[command]}
        for command, target in _workflow_script_commands().items()
        if command in COMMAND_MODULES and target.split(":", 1)[0] != COMMAND_MODULES[command]
    }

    assert not mismatched


def test_json_command_listing_covers_all_grouped_commands() -> None:
    """Keep the machine-readable CLI inventory complete and internally consistent."""

    payload = json.loads(_command_listing("json"))
    records = payload["commands"]
    listed = {record["command"]: record for record in records}

    assert set(listed) == set(COMMAND_MODULES)
    assert [record["command"] for record in records] == sorted(COMMAND_MODULES)
    for command, record in listed.items():
        assert record["module"] == COMMAND_MODULES[command]
        assert command not in record["aliases"]
        for alias in record["aliases"]:
            assert COMMAND_MODULES[alias] == record["module"]
