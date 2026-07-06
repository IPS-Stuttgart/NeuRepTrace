from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "neureptrace._stimulus_detection_public",
        "neureptrace.stimulus_detection",
        "neureptrace.event_detection",
    ],
)
def test_detection_cli_mains_accept_explicit_argv(module_name: str, capsys) -> None:
    module = importlib.import_module(module_name)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--conflict-resolution" in help_text
    assert "--out-events" in help_text
