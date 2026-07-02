from __future__ import annotations

import runpy

import pytest


def test_results_package_module_entrypoint_propagates_main_exit_code(monkeypatch):
    import neureptrace.results as results_module

    calls = []

    def fake_main() -> int:
        calls.append(True)
        return 13

    monkeypatch.setattr(results_module, "main", fake_main)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("neureptrace.results", run_name="__main__")

    assert exc_info.value.code == 13
    assert calls == [True]
