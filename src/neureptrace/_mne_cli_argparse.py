from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def run_with_conflict_resolving_parser(parser_owner: Any, main: Callable[[], T]) -> T:
    argument_parser = parser_owner.argparse.ArgumentParser
    original_init = argument_parser.__init__

    def conflict_resolving_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("conflict_handler", "resolve")
        original_init(self, *args, **kwargs)

    argument_parser.__init__ = conflict_resolving_init
    try:
        return main()
    finally:
        argument_parser.__init__ = original_init
