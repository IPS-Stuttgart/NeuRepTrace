import sys
from importlib import import_module
from pathlib import Path
import runpy

import pytest
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


def test_package_module_entrypoint_delegates_to_grouped_cli(monkeypatch):
    import neureptrace.cli as cli_module

    calls = []

    def fake_main():
        calls.append(True)
        return 7

    monkeypatch.setattr(cli_module, "main", fake_main)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("neureptrace.__main__", run_name="__main__")

    assert exc_info.value.code == 7
    assert calls == [True]


def test_observation_ensemble_package_module_entrypoint_delegates(monkeypatch):
    import neureptrace.observation_ensemble as observation_ensemble_module

    calls = []

    def fake_main():
        calls.append(True)
        return 13

    monkeypatch.setattr(observation_ensemble_module, "main", fake_main)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("neureptrace.observation_ensemble", run_name="__main__")

    assert exc_info.value.code == 13
    assert calls == [True]


def test_fieldtrip_module_entrypoint_exposes_path_options(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["python -m neureptrace.fieldtrip_mat", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("neureptrace.fieldtrip_mat", run_name="__main__")

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--trial-path",
        "--time-path",
        "--label-path",
        "--trialinfo-path",
        "--sampleinfo-path",
    ):
        assert option in help_text


def test_mne_time_decode_scripts_use_safe_wrappers():
    scripts = _poetry_scripts()

    assert scripts["neureptrace-mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli:main"
    assert scripts["neureptrace-mne-time-decode-base"] == "neureptrace.mne_time_decode_cli:main"
    assert COMMAND_MODULES["mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli"
    assert COMMAND_MODULES["mne-time-decode-base"] == "neureptrace.mne_time_decode_cli"


def test_openneuro_resilient_entry_points_are_exposed():
    scripts = _poetry_scripts()

    assert scripts["neureptrace-openneuro-resilient"] == "neureptrace.openneuro_resilient:main"
    assert COMMAND_MODULES["openneuro-resilient"] == "neureptrace.openneuro_resilient"


def test_emission_compare_entry_points_are_exposed():
    scripts = _poetry_scripts()

    assert scripts["neureptrace-emission-compare"] == "neureptrace.emission_compare:main"
    assert COMMAND_MODULES["emission-compare"] == "neureptrace.emission_compare"


def test_response_window_ensemble_entry_points_are_exposed():
    scripts = _poetry_scripts()

    assert scripts["neureptrace-response-window-ensemble"] == "neureptrace.response_window_ensemble:main"
    assert COMMAND_MODULES["response-window-ensemble"] == "neureptrace.response_window_ensemble"


def test_bushmeg_category2_autoencoder_entry_points_are_exposed():
    scripts = _poetry_scripts()

    assert scripts["neureptrace-bushmeg-category2-autoencoder-loso"] == "neureptrace.bushmeg_category2_autoencoder_loso:main"
    assert COMMAND_MODULES["bushmeg-category2-autoencoder-loso"] == "neureptrace.bushmeg_category2_autoencoder_loso"
