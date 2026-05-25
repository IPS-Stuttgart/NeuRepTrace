from __future__ import annotations

import sys
import tomllib
import types
from pathlib import Path

from neureptrace import cli


def test_grouped_cli_dispatches_to_module_main(monkeypatch):
    calls = []
    fake_module = types.ModuleType("fake_neureptrace_command")

    def fake_main():
        calls.append(tuple(sys.argv))
        return None

    fake_module.main = fake_main
    monkeypatch.setitem(sys.modules, "fake_neureptrace_command", fake_module)
    monkeypatch.setitem(cli.COMMAND_MODULES, "fake", "fake_neureptrace_command")

    assert cli.main(["fake", "--value", "42"]) == 0
    assert calls == [("neureptrace fake", "--value", "42")]


def test_grouped_cli_exposes_mne_decoder_variants():
    assert cli.COMMAND_MODULES["mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli"
    assert cli.COMMAND_MODULES["mne-time-decode-base"] == "neureptrace.mne_time_decode_cli"
    assert cli.COMMAND_MODULES["mne-time-decode-ensemble"] == "neureptrace.mne_time_decode_ensemble"


def test_grouped_cli_exposes_synthetic_fieldtrip_generator():
    assert cli.COMMAND_MODULES["synthetic-fieldtrip"] == "neureptrace.synthetic_fieldtrip"


def test_poetry_scripts_expose_mne_decoder_variants():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["tool"]["poetry"]["scripts"]

    assert scripts["neureptrace-mne-time-decode"] == "neureptrace.mne_time_decode_foldlocal_cli:main"
    assert scripts["neureptrace-mne-time-decode-base"] == "neureptrace.mne_time_decode_cli:main"
    assert scripts["neureptrace-mne-time-decode-ensemble"] == "neureptrace.mne_time_decode_ensemble:main"


def test_primary_grouped_cli_commands_have_direct_console_scripts():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["tool"]["poetry"]["scripts"]

    grouped_only_aliases = {
        "dataset-config",
        "ensemble-observations",
        "event-detection",
        "mne-transfer-decode",
        "observation-schema",
        "onset-detection",
        "stimulus-detection",
        "transfer-decode",
    }
    missing = []
    mismatched = []
    for command, module in sorted(cli.COMMAND_MODULES.items()):
        if command in grouped_only_aliases:
            continue
        script_name = f"neureptrace-{command}"
        expected_target = f"{module}:main"
        actual_target = scripts.get(script_name)
        if actual_target is None:
            missing.append(script_name)
        elif actual_target != expected_target:
            mismatched.append((script_name, expected_target, actual_target))

    assert not missing
    assert not mismatched


def test_grouped_cli_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    captured = capsys.readouterr()
    assert "NeuRepTrace command-line interface" in captured.out


def test_grouped_cli_help_without_command_lists_commands(capsys):
    assert cli.main(["help"]) == 0
    captured = capsys.readouterr()

    assert "Available commands:" in captured.out
    assert "benchmark" in captured.out
    assert "neureptrace.benchmark" in captured.out


def test_grouped_cli_help_subcommand_dispatches_module_help(monkeypatch):
    calls = []
    fake_module = types.ModuleType("fake_neureptrace_help_command")

    def fake_main():
        calls.append(tuple(sys.argv))
        return None

    fake_module.main = fake_main
    monkeypatch.setitem(sys.modules, "fake_neureptrace_help_command", fake_module)
    monkeypatch.setitem(cli.COMMAND_MODULES, "fake", "fake_neureptrace_help_command")

    assert cli.main(["help", "fake"]) == 0
    assert calls == [("neureptrace fake", "--help")]


def test_grouped_cli_help_unknown_command_suggests_close_match(capsys):
    try:
        cli.main(["help", "benchmrk"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - the command must exit with an argparse-compatible failure
        raise AssertionError("Unknown help target did not fail")

    captured = capsys.readouterr()
    assert "unknown command 'benchmrk'" in captured.err
    assert "benchmark" in captured.err
