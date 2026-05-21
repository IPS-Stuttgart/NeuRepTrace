"""Argparse compatibility helpers for MNE time-decoding entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def run_with_conflict_resolving_parser(parser_owner: Any, main: Callable[[], T]) -> T:
    """Run ``main`` while duplicate argparse options are resolved locally.

    The historical ``neureptrace.mne_time_decode`` parser contains duplicate
    FieldTrip options.  Importing the module is still useful because downstream
    modules reuse its helper functions, but constructing the parser should not
    make installed console entry points unusable.  This wrapper keeps the
    compatibility behavior scoped to the MNE CLI call and restores argparse
    afterwards, including when ``--help`` raises ``SystemExit``.
    """

    original_argument_parser = parser_owner.argparse.ArgumentParser

    class ConflictResolvingArgumentParser(original_argument_parser):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("conflict_handler", "resolve")
            super().__init__(*args, **kwargs)

    parser_owner.argparse.ArgumentParser = ConflictResolvingArgumentParser
    try:
        return main()
    finally:
        parser_owner.argparse.ArgumentParser = original_argument_parser
