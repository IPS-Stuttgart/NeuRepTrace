from importlib import import_module
from pathlib import Path

import tomllib

from neureptrace.cli import COMMAND_MODULES


def _poetry_scripts() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["poetry"]["scripts"]


def test_poetry_console_script_targets_are_importable():
    for script_name, target in _poetry_scripts().items():
        module_name, function_name = target.split(":", maxsplit=1)
        module = import_module(module_name)

        assert callable(getattr(module, function_name)), script_name


def test_grouped_cli_targets_are_importable():
    for command, module_name in COMMAND_MODULES.items():
        module = import_module(module_name)

        assert callable(getattr(module, "main", None)), command


def test_mne_time_decode_scripts_use_safe_wrappers():
    scripts = _poetry_scripts()

    assert scripts["neureptrace-mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli:main"
    assert scripts["neureptrace-mne-time-decode-base"] == "neureptrace.mne_time_decode_cli:main"
    assert COMMAND_MODULES["mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli"
    assert COMMAND_MODULES["mne-time-decode-base"] == "neureptrace.mne_time_decode_cli"
