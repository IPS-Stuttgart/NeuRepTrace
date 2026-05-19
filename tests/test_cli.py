from __future__ import annotations

import sys
import types

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


def test_grouped_cli_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    captured = capsys.readouterr()
    assert "NeuRepTrace command-line interface" in captured.out
