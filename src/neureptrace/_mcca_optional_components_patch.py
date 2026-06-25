"""Normalize optional M-CCA subject PCA component settings.

The source-alignment and category-2 calibration config builders both expose
``mcca_subject_pca_components`` as an optional nested dimensionality-reduction
setting.  Config files and command-line wrappers may pass empty, ``none``, or
``null`` strings for disabled optional settings; these should behave like
``None`` rather than being parsed as an explicit component request.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_mcca_optional_components_patch_installed"
_DISABLED_TEXT_VALUES = {"", "none", "null"}


def _normalize_optional_components_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _DISABLED_TEXT_VALUES:
        return None
    return value


def _wrap_config_function(function: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(function, _PATCH_MARKER, False):
        return function

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if "mcca_subject_pca_components" in kwargs:
            kwargs = dict(kwargs)
            kwargs["mcca_subject_pca_components"] = _normalize_optional_components_value(
                kwargs["mcca_subject_pca_components"]
            )
        return function(*args, **kwargs)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _patch(module_name: str, function_name: str) -> None:
    module = importlib.import_module(module_name)
    setattr(module, function_name, _wrap_config_function(getattr(module, function_name)))


def install() -> None:
    """Install optional M-CCA component normalization at config boundaries."""

    _patch("neureptrace.decoding.source_alignment", "source_alignment_config")
    _patch("neureptrace.decoding.unlabeled_calibration_alignment", "unlabeled_calibration_alignment_config")


__all__ = ["install"]
