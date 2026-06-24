"""Runtime patch for Protocol 3 BUSH-MEG subject-loader compatibility."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from typing import Any, Callable


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def install() -> None:
    """Install a Protocol 3 subject-loader wrapper that honors older loader signatures."""
    import neureptrace.bushmeg_all_protocols as all_protocols
    import neureptrace.bushmeg_source_loso as source_loso

    if getattr(all_protocols._load_protocol3_subjects_cached, "_subject_loader_signature_patched", False):
        return

    def _load_protocol3_subjects_cached(
        config: Mapping[str, Any],
        *,
        config_dir: Path,
        progress_callback: Callable[..., None] | None,
    ) -> tuple[Any, Any]:
        key = all_protocols._protocol3_subject_cache_key(config, config_dir=config_dir)
        if key in all_protocols._PROTOCOL3_SUBJECT_CACHE:
            subjects, encoder = all_protocols._PROTOCOL3_SUBJECT_CACHE[key]
            if progress_callback is not None:
                progress_callback("loading_subjects", cache_hit=True, n_subject_files=len(subjects))
            return subjects, encoder

        loader = source_loso._load_subjects_from_config
        kwargs: dict[str, Any] = {"config_dir": config_dir}
        if _accepts_keyword(loader, "progress_callback"):
            kwargs["progress_callback"] = progress_callback
        subjects, encoder = loader(config, **kwargs)
        all_protocols._PROTOCOL3_SUBJECT_CACHE[key] = (subjects, encoder)
        return subjects, encoder

    _load_protocol3_subjects_cached._subject_loader_signature_patched = True  # type: ignore[attr-defined]
    all_protocols._load_protocol3_subjects_cached = _load_protocol3_subjects_cached


__all__ = ["install"]
