"""Fail fast for unsupported source-time selection option combinations."""

from __future__ import annotations

from functools import wraps

_PATCH_MARKER = "_neureptrace_mne_source_time_selection_guard_patch_installed"
_DISABLED = {"", "none", "off", "false", "0", "raw"}
_FALSE_VALUES = {"", "0", "false", "f", "no", "n", "off", "none", "null"}
_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_DANN_ALIASES = {"dann", "domain_adversarial", "domain_adversarial_nn", "domain_adversarial_neural_network"}


def _token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _enabled(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    return _token(value) not in _DISABLED


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        token = _token(value)
        if token in _FALSE_VALUES:
            return False
        if token in _TRUE_VALUES:
            return True
    return bool(value)


def install() -> None:
    import neureptrace.mne_time_decode as mne_time_decode

    original = mne_time_decode.run_time_resolved_decode
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def run_time_resolved_decode(*args, **kwargs):
        if _enabled(kwargs.get("source_time_selection", "none")):
            if _enabled(kwargs.get("alignment_method", "none")):
                raise ValueError("source_time_selection cannot be combined with source alignment; set alignment_method='none'.")
            if _truthy(kwargs.get("pseudo_label_self_training", False)):
                raise ValueError("source_time_selection cannot be combined with pseudo_label_self_training.")
            if _token(kwargs.get("decoder", "logistic")) in _DANN_ALIASES:
                raise ValueError("source_time_selection cannot be combined with the DANN decoder.")
        return original(*args, **kwargs)

    setattr(run_time_resolved_decode, _PATCH_MARKER, True)
    mne_time_decode.run_time_resolved_decode = run_time_resolved_decode


__all__ = ["install"]
