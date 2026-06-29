"""Validate v1 dataset-spec role references during file resolution.

The v1 spec validator already reports roles whose ``file_role`` is missing from
``participants.files``.  Direct file-table/listing calls go through
``resolve_dataset_files`` and previously indexed ``files[file_role]`` directly,
which exposed a raw ``KeyError`` instead of the actionable validation message.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_PATCH_MARKER = "_neureptrace_dataset_spec_role_reference_patch_installed"


def _missing_file_role_message(role: Any, file_role: Any) -> str:
    return f"roles.{role}.file_role='{file_role}' is not defined in participants.files"


def install() -> None:
    """Install role-reference validation for v1 dataset-spec file resolution."""

    from neureptrace.datasets import spec as dataset_spec

    if getattr(dataset_spec, _PATCH_MARKER, False):
        return

    original_role_to_file_roles = dataset_spec._role_to_file_roles

    def _role_to_file_roles(spec: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, str]:
        role_to_file_role = original_role_to_file_roles(spec, files)
        for role, file_role in role_to_file_role.items():
            if file_role not in files:
                raise ValueError(_missing_file_role_message(role, file_role))
        return role_to_file_role

    _role_to_file_roles.__doc__ = original_role_to_file_roles.__doc__
    dataset_spec._role_to_file_roles = _role_to_file_roles
    setattr(dataset_spec, _PATCH_MARKER, True)


__all__ = ["install"]
