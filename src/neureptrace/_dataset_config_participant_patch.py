"""Runtime compatibility patch for strict dataset participant IDs.

The dataset-config parser accepts compact participant specifications such as
``1-4,6,sub-09``.  Python booleans are subclasses of ``int`` and mappings are
iterable over their keys, so malformed YAML/JSON snippets such as
``participants: {ids: true}`` or ``participants: {ids: {subject: 1}}`` can be
silently interpreted as participant tokens.  This patch keeps the public parser
API stable while rejecting those ambiguous inputs before path templates are
expanded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_PATCH_MARKER = "_neureptrace_participant_id_patch_installed"


def _validate_participant_token(token: Any) -> None:
    if isinstance(token, bool):
        raise ValueError("participants.ids entries must be participant identifiers, not booleans.")
    if isinstance(token, Mapping):
        raise ValueError("participants.ids entries must be scalars, not mappings.")
    if isinstance(token, str) and token.strip().lower() in {"true", "false", "yes", "no"}:
        raise ValueError("participants.ids entries must be participant identifiers, not booleans.")


def _validate_participant_ids_input(value: Any) -> None:
    if isinstance(value, bool):
        raise ValueError("participants.ids must be an int, string, or list, not a boolean.")
    if isinstance(value, Mapping):
        raise ValueError("participants.ids must be an int, string, or list, not a mapping.")
    if isinstance(value, str):
        return
    if isinstance(value, Iterable):
        for token in value:
            _validate_participant_token(token)


def install() -> None:
    """Install strict participant-ID input validation."""

    from neureptrace import dataset_config

    if getattr(dataset_config, _PATCH_MARKER, False):
        return

    original_parse_participant_ids = dataset_config.parse_participant_ids

    def parse_participant_ids(value: Any) -> list[int | str]:
        _validate_participant_ids_input(value)
        parsed = original_parse_participant_ids(value)
        for token in parsed:
            _validate_participant_token(token)
        return parsed

    parse_participant_ids.__doc__ = original_parse_participant_ids.__doc__
    dataset_config.parse_participant_ids = parse_participant_ids
    setattr(dataset_config, _PATCH_MARKER, True)
