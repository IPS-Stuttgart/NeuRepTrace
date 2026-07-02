"""FieldTrip config-level participant metadata support."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MARKER = "_neureptrace_io_fieldtrip_config_participant_installed"


def install() -> None:
    import neureptrace.io.fieldtrip_mat as fieldtrip_mat

    if getattr(fieldtrip_mat.load_fieldtrip_mat_epochs, _MARKER, False):
        return

    original = fieldtrip_mat.load_fieldtrip_mat_epochs

    def load_fieldtrip_mat_epochs(
        path: str | Path,
        config: Mapping[str, Any] | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        config_dict = dict(config or {})
        metadata = dict(extra_metadata or {})
        if "participant" in config_dict and "participant" not in metadata:
            participant = config_dict["participant"]
            if participant is not None:
                metadata["participant"] = str(participant)
        return original(path, config_dict, extra_metadata=metadata)

    setattr(load_fieldtrip_mat_epochs, _MARKER, True)
    fieldtrip_mat.load_fieldtrip_mat_epochs = load_fieldtrip_mat_epochs


__all__ = ["install"]
