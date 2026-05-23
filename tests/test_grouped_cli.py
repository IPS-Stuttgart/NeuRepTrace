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


def test_grouped_cli_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert f"neureptrace {__version__}" in capsys.readouterr().out


def test_grouped_bushmeg_data_help(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["neureptrace"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["bushmeg-data", "--help"])

    assert exc_info.value.code == 0
    assert sys.argv == ["neureptrace"]
    assert "prepare-smoke" in capsys.readouterr().out
