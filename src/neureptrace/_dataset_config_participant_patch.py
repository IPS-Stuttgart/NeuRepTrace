"""Runtime compatibility patch for dataset participant IDs and templates.

The dataset-config parser accepts compact participant specifications such as
``1-4,6,sub-09``.  Python booleans are subclasses of ``int`` and mappings are
iterable over their keys, so malformed YAML/JSON snippets such as
``participants: {ids: true}`` or ``participants: {ids: {subject: 1}}`` can be
silently interpreted as participant tokens.  This patch keeps the public parser
API stable while rejecting those ambiguous inputs before path templates are
expanded.

It also keeps single FieldTrip participant templates aligned with the documented
multi-template and MNE template formatting vocabulary.  Paths such as
``Part{participant02d}Data.mat`` should resolve the same way whether they are
configured through ``dataset.file_template``/``dataset.participant_file`` or
through ``dataset.file_templates``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
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
        _validate_participant_ids_input(value)
        parsed = original_parse_participant_ids(value)
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
