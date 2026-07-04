"""Runtime compatibility patch for dataset participant IDs and templates.

The dataset-config parser accepts compact participant specifications such as
``1-4,6,sub-09``.  Python booleans are subclasses of ``int`` and mappings are
iterable over their keys, so malformed YAML/JSON snippets such as
``participants: {ids: true}`` or ``participants: {ids: {subject: 1}}`` can be
silently interpreted as participant tokens.  This patch keeps the public parser
API stable while rejecting those ambiguous inputs before path templates are
expanded.

Programmatic dataset specs may also pass NumPy/Pandas integral scalar values
for ``participants.ids``.  Those values should behave like plain Python integer
IDs, while NumPy boolean scalars should still be rejected as booleans rather
than being treated as participant identifiers.

It also keeps single FieldTrip participant templates aligned with the documented
multi-template and MNE template formatting vocabulary.  Paths such as
``Part{participant02d}Data.mat`` should resolve the same way whether they are
configured through ``dataset.file_template``/``dataset.participant_file`` or
through ``dataset.file_templates``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import wraps
from numbers import Integral
from typing import Any

_PATCH_MARKER = "_neureptrace_participant_id_patch_installed"
_SIGNED_INT_RANGE_RE = re.compile(r"^([+-]?\d+)-([+-]?\d+)$")
_BOOLEAN_PARTICIPANT_TEXT = {"true", "false", "yes", "no"}


def _is_boolean_scalar(value: Any) -> bool:
    return isinstance(value, bool) or (type(value).__module__ == "numpy" and type(value).__name__ in {"bool", "bool_"})


def _is_integral_identifier(value: Any) -> bool:
    return isinstance(value, Integral) and not _is_boolean_scalar(value)


def _validate_participant_token(token: Any) -> None:
    if _is_boolean_scalar(token):
        raise ValueError("participants.ids entries must be participant identifiers, not booleans.")
    if isinstance(token, Mapping):
        raise ValueError("participants.ids entries must be scalars, not mappings.")
    if isinstance(token, str) and token.strip().lower() in _BOOLEAN_PARTICIPANT_TEXT:
        raise ValueError("participants.ids entries must be participant identifiers, not booleans.")


def _validate_participant_ids_input(value: Any) -> None:
    if _is_boolean_scalar(value):
        raise ValueError("participants.ids must be an int, string, or list, not a boolean.")
    if _is_integral_identifier(value):
        return
    if isinstance(value, Mapping):
        raise ValueError("participants.ids must be an int, string, or list, not a mapping.")
    if isinstance(value, str):
        return
    if isinstance(value, Iterable):
        for token in value:
            _validate_participant_token(token)


def _expand_participant_token(token: Any) -> list[int | str]:
    """Expand one participant token with signed integer/range support."""

    _validate_participant_token(token)
    if _is_integral_identifier(token):
        return [int(token)]

    text = str(token).strip()
    if not text:
        return []
    if "," in text:
        expanded: list[int | str] = []
        for part in text.split(","):
            expanded.extend(_expand_participant_token(part))
        return expanded
    if text.lower() in _BOOLEAN_PARTICIPANT_TEXT:
        raise ValueError("participants.ids entries must be participant identifiers, not booleans.")

    range_match = _SIGNED_INT_RANGE_RE.fullmatch(text)
    if range_match:
        start = int(range_match.group(1))
        stop = int(range_match.group(2))
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))

    try:
        return [int(text)]
    except ValueError:
        return [text]


def _parse_participant_ids(value: Any) -> list[int | str]:
    """Parse compact participant specifications without confusing signs for ranges."""

    if value is None:
        return []
    if _is_integral_identifier(value):
        return [int(value)]
    _validate_participant_ids_input(value)
    if isinstance(value, str):
        return _expand_participant_token(value)
    if isinstance(value, Iterable):
        parsed: list[int | str] = []
        for token in value:
            parsed.extend(_expand_participant_token(token))
        return parsed
    raise ValueError("participants.ids must be an int, string, or list.")


def _single_fieldtrip_template_context(dataset_config: Any, config: Mapping[str, Any]):
    """Return the single FieldTrip participant template context, if present."""

    dataset = dataset_config._dataset_section(config)
    if dataset.get("type") != "fieldtrip_mat":
        return None
    if dataset.get("files"):
        return None
    if dataset_config._participant_file_templates(dataset):
        return None
    template = dataset.get("participant_file") or dataset.get("file_template")
    if template is None:
        return None
    return dataset, str(template)


def _format_single_participant_path(dataset_config: Any, *, template: str, participant: int | str, base_dir: Any, root: Any):
    format_values = dataset_config._format_values_for_participant(participant)
    return dataset_config.expand_path(template.format(**format_values), base_dir=base_dir, root=root)


def install() -> None:
    """Install strict participant-ID input validation and template formatting."""

    from neureptrace import dataset_config

    if getattr(dataset_config, _PATCH_MARKER, False):
        return

    original_parse_participant_ids = dataset_config.parse_participant_ids
    original_iter_dataset_files = dataset_config.iter_dataset_files
    original_fieldtrip_file_specs = dataset_config._fieldtrip_file_specs

    @wraps(original_parse_participant_ids)
    def parse_participant_ids(value: Any) -> list[int | str]:
        parsed = _parse_participant_ids(value)
        for token in parsed:
            _validate_participant_token(token)
        return parsed

    @wraps(original_iter_dataset_files)
    def iter_dataset_files(config: Mapping[str, Any], *, base_dir: Any = "."):
        context = _single_fieldtrip_template_context(dataset_config, config)
        if context is None:
            return original_iter_dataset_files(config, base_dir=base_dir)
        dataset, template = context
        root = dataset.get("root")
        return [
            _format_single_participant_path(
                dataset_config,
                template=template,
                participant=participant,
                base_dir=base_dir,
                root=root,
            )
            for participant in dataset_config._participant_ids(config)
        ]

    @wraps(original_fieldtrip_file_specs)
    def _fieldtrip_file_specs(config: Mapping[str, Any], *, base_dir: Any):
        context = _single_fieldtrip_template_context(dataset_config, config)
        if context is None:
            return original_fieldtrip_file_specs(config, base_dir=base_dir)
        dataset, template = context
        root = dataset.get("root")
        specs = []
        for participant in dataset_config._participant_ids(config):
            path = _format_single_participant_path(
                dataset_config,
                template=template,
                participant=participant,
                base_dir=base_dir,
                root=root,
            )
            specs.append((path, {"participant": participant}))
        return specs

    dataset_config.parse_participant_ids = parse_participant_ids
    dataset_config.iter_dataset_files = iter_dataset_files
    dataset_config._fieldtrip_file_specs = _fieldtrip_file_specs
    setattr(dataset_config, _PATCH_MARKER, True)
