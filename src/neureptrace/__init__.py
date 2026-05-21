"""Probabilistic tracing of neural representations over time."""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"

from neureptrace import _decoding_regularization_patch, _event_detection_extensions

_event_detection_extensions.install()
_decoding_regularization_patch.install()
