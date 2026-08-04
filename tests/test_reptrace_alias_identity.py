from __future__ import annotations

import importlib
import sys

import pytest


_ALIAS_NAME = "reptrace.metrics.ranking"
_TARGET_NAME = "neureptrace.metrics.ranking"


def test_reptrace_submodule_alias_uses_canonical_module_object() -> None:
    alias = importlib.import_module(_ALIAS_NAME)
    target = importlib.import_module(_TARGET_NAME)

    assert alias is target
    assert sys.modules[_ALIAS_NAME] is target


def test_reptrace_monkeypatch_updates_function_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    alias = importlib.import_module(_ALIAS_NAME)

    def fail_from_alias(*_args, **_kwargs):
        raise RuntimeError("legacy alias reached canonical globals")

    monkeypatch.setattr(alias, "_validate_integer", fail_from_alias)

    with pytest.raises(RuntimeError, match="legacy alias reached canonical globals"):
        alias.rank_class_scores(None, None, [], top_k=(1,))
