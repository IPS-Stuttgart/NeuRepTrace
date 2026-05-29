from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def run_with_conflict_resolving_parser(parser_owner: Any, main: Callable[[], T]) -> T:
    """Run a legacy CLI with duplicate-option resolution, then restore argparse.

    The base MNE time-decode parser still carries compatibility options that are
    registered by more than one wrapper.  Keep argparse's conflict-resolution
    workaround scoped to the wrapped command, including nested wrapper calls and
    early ``SystemExit`` paths from ``--help``.
    """

    argument_parser = parser_owner.argparse.ArgumentParser
    original_init = argument_parser.__init__

    if getattr(original_init, "_neureptrace_conflict_resolving", False):
        return main()

    def conflict_resolving_init(self: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("conflict_handler", "resolve")
        original_init(self, *args, **kwargs)

    conflict_resolving_init._neureptrace_conflict_resolving = True  # type: ignore[attr-defined]
    argument_parser.__init__ = conflict_resolving_init
    try:
        return main()
    finally:
        if argument_parser.__init__ is conflict_resolving_init:
            argument_parser.__init__ = original_init
