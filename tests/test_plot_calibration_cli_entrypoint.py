from __future__ import annotations

import tomllib
from pathlib import Path


def test_plot_calibration_has_console_script_entrypoint() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    scripts = pyproject["tool"]["poetry"]["scripts"]

    assert scripts["neureptrace-plot-calibration"] == "neureptrace.plot_calibration:main"


def test_plot_calibration_console_script_target_imports() -> None:
    from neureptrace.plot_calibration import main

    assert callable(main)


def test_plot_calibration_is_available_from_grouped_cli() -> None:
    from neureptrace.cli import COMMAND_MODULES

    assert COMMAND_MODULES["plot-calibration"] == "neureptrace.plot_calibration"
