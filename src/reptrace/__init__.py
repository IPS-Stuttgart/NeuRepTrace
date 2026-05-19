"""Backward-compatible ``reptrace`` namespace for legacy downstream users.

The project was renamed to :mod:`neureptrace`, but PyMEGDec and older scripts
still import :mod:`reptrace`.  Keep those imports working by resolving
``reptrace.*`` modules from the corresponding ``neureptrace.*`` modules.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from typing import Any

_ALIAS_PREFIX = __name__ + "."
_TARGET_PREFIX = "neureptrace."
_FINDER_MARKER = "_reptrace_alias_finder"


class _ReptraceAliasLoader(importlib.abc.Loader):
    """Load ``reptrace`` submodules from the matching ``neureptrace`` module."""

    def __init__(self, alias_name: str, target_name: str) -> None:
        self.alias_name = alias_name
        self.target_name = target_name

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        target = importlib.import_module(self.target_name)
        alias_spec = module.__spec__
        alias_loader = module.__loader__
        module.__dict__.update(target.__dict__)
        module.__name__ = self.alias_name
        module.__package__ = self.alias_name.rpartition(".")[0]
        module.__loader__ = alias_loader
        module.__spec__ = alias_spec
        if hasattr(target, "__path__"):
            module.__path__ = []


class _ReptraceAliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``reptrace.*`` import specs against ``neureptrace.*``."""

    _reptrace_alias_finder = True

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        if not fullname.startswith(_ALIAS_PREFIX):
            return None
        target_name = _TARGET_PREFIX + fullname[len(_ALIAS_PREFIX) :]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None
        is_package = target_spec.submodule_search_locations is not None
        spec = importlib.util.spec_from_loader(
            fullname,
            _ReptraceAliasLoader(fullname, target_name),
            origin=getattr(target_spec, "origin", None),
            is_package=is_package,
        )
        if spec is not None and is_package:
            spec.submodule_search_locations = []
        return spec


def _install_alias_finder() -> None:
    if not any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ReptraceAliasFinder())


_target_module = importlib.import_module("neureptrace")
__all__ = getattr(_target_module, "__all__", ())
__version__ = getattr(_target_module, "__version__", "0.0.0")
__path__: list[str] = []
_install_alias_finder()


def __getattr__(name: str) -> Any:
    return getattr(_target_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_target_module)))
