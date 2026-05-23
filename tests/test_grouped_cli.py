from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from neureptrace import __version__, cli


def _console_scripts() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["poetry"]["scripts"]


def test_grouped_cli_exposes_focused_console_scripts() -> None:
    """Every focused ``neureptrace-*`` script should have a grouped alias."""

    scripts = _console_scripts()
    expected_commands = {
        script_name.removeprefix("neureptrace-")
        for script_name in scripts
        if script_name.startswith("neureptrace-")
    }

    assert expected_commands <= set(cli.COMMAND_MODULES)


def test_grouped_cli_aliases_match_focused_console_script_targets() -> None:
    """Grouped aliases should dispatch to the same modules as focused scripts."""

    scripts = _console_scripts()
    for script_name, target in scripts.items():
        if script_name == "neureptrace" or not script_name.startswith("neureptrace-"):
            continue
        command = script_name.removeprefix("neureptrace-")
        module_name, function_name = target.split(":", maxsplit=1)

        assert function_name == "main", script_name
        assert cli.COMMAND_MODULES[command] == module_name, script_name


def test_temporal_smoothing_grouped_alias_matches_console_script() -> None:
    """Keep the grouped temporal-smoothing command aligned with its focused script."""

    assert _console_scripts()["neureptrace-temporal-smoothing"] == "neureptrace.temporal_smoothing:main"
    assert cli.COMMAND_MODULES["temporal-smoothing"] == "neureptrace.temporal_smoothing"


def test_grouped_cli_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert f"neureptrace {__version__}" in capsys.readouterr().out


def test_grouped_cli_unknown_command_suggests_close_match(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["stimulus-deteckt", "--help"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "unknown command 'stimulus-deteckt'" in stderr
    assert "stimulus-detect" in stderr
    assert "--list-commands" in stderr


def test_grouped_cli_unknown_option_without_command_is_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--not-a-neureptrace-option"])

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --not-a-neureptrace-option" in capsys.readouterr().err


def test_grouped_bushmeg_data_help(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["neureptrace"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["bushmeg-data", "--help"])

    assert exc_info.value.code == 0
    assert sys.argv == ["neureptrace"]
    assert "prepare-smoke" in capsys.readouterr().out
