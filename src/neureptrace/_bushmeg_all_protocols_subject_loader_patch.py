"""Runtime patch for Protocol 3 BUSH-MEG subject-loader compatibility."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from typing import Any, Callable

_SOURCE_LOADER_MARKER = "_subject_loader_progress_callback_patched"
_PROTOCOL3_CACHE_MARKER = "_subject_loader_signature_patched"


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _patch_source_loader(source_loso: Any) -> None:
    """Make the shared BUSH-MEG subject loader tolerant to optional progress callbacks."""

    loader = source_loso._load_subjects_from_config
    if getattr(loader, _SOURCE_LOADER_MARKER, False):
        return

    def _load_subjects_from_config(
        config: Mapping[str, Any],
        *,
        config_dir: Path,
        progress_callback: Callable[..., None] | None = None,
    ) -> tuple[Any, Any]:
        kwargs: dict[str, Any] = {"config_dir": config_dir}
        if _accepts_keyword(loader, "progress_callback"):
            kwargs["progress_callback"] = progress_callback
        return loader(config, **kwargs)

    setattr(_load_subjects_from_config, _SOURCE_LOADER_MARKER, True)
    setattr(_load_subjects_from_config, "__wrapped__", loader)
    source_loso._load_subjects_from_config = _load_subjects_from_config


def install() -> None:
    """Install subject-loader wrappers that honor older loader signatures."""
    import neureptrace.bushmeg_all_protocols as all_protocols
    import neureptrace.bushmeg_source_loso as source_loso

    _patch_source_loader(source_loso)

    if getattr(all_protocols._load_protocol3_subjects_cached, _PROTOCOL3_CACHE_MARKER, False):
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

    setattr(_load_protocol3_subjects_cached, _PROTOCOL3_CACHE_MARKER, True)
    all_protocols._load_protocol3_subjects_cached = _load_protocol3_subjects_cached


__all__ = ["install"]
