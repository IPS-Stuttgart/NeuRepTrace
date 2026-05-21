from __future__ import annotations

import tomllib
import types
from pathlib import Path

from neureptrace import cli


def test_grouped_cli_dispatches_to_module_main(monkeypatch):
    calls = []
    module_name = "fake_neureptrace_command"
    command_name = "fake"

    def fake_main():
        calls.append(tuple(cli.sys.argv))
        return None

    def fake_import_module(name: str):
        assert name == module_name
        return types.SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli, "import_module", fake_import_module)
    monkeypatch.setitem(cli.COMMAND_MODULES, command_name, module_name)

    assert cli.main([command_name, "--value", "42"]) == 0
    assert calls == [("neureptrace fake", "--value", "42")]


def test_grouped_cli_exposes_mne_decoder_variants():
    assert cli.COMMAND_MODULES["mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli"
    assert cli.COMMAND_MODULES["mne-time-decode-base"] == "neureptrace.mne_time_decode_cli"
    assert cli.COMMAND_MODULES["mne-time-decode-ensemble"] == "neureptrace.mne_time_decode_ensemble"


def test_poetry_scripts_expose_mne_decoder_variants():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["tool"]["poetry"]["scripts"]

    assert scripts["neureptrace-mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli:main"
    assert scripts["neureptrace-mne-time-decode-base"] == "neureptrace.mne_time_decode_cli:main"
    assert scripts["neureptrace-mne-time-decode-ensemble"] == "neureptrace.mne_time_decode_ensemble:main"


def test_grouped_cli_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    captured = capsys.readouterr()
    assert "NeuRepTrace command-line interface" in captured.out
