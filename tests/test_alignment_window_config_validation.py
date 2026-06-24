from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from neureptrace.decoding.alignment_window import resolved_alignment_window, uses_separate_alignment_window


@dataclass(frozen=True)
class DummyConfig:
    window_center: object = 0.3
    window_size: object = 0.1
    alignment_window_center: object | None = None
    alignment_window_size: object | None = None


@pytest.mark.parametrize("center", [True, np.bool_(False), np.nan, np.inf, [0.1]])
def test_resolved_alignment_window_rejects_invalid_alignment_center(center: object) -> None:
    with pytest.raises(ValueError, match="alignment_window_center"):
        resolved_alignment_window(DummyConfig(alignment_window_center=center))


@pytest.mark.parametrize("size", [True, np.bool_(False), 0.0, -0.1, np.nan, np.inf, [0.1]])
def test_resolved_alignment_window_rejects_invalid_alignment_size(size: object) -> None:
    with pytest.raises(ValueError, match="alignment_window_size"):
        resolved_alignment_window(DummyConfig(alignment_window_size=size))


def test_resolved_alignment_window_rejects_invalid_default_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        resolved_alignment_window(DummyConfig(window_size=False))


def test_uses_separate_alignment_window_rejects_invalid_alignment_size() -> None:
    with pytest.raises(ValueError, match="alignment_window_size"):
        uses_separate_alignment_window(DummyConfig(alignment_window_size=0.0))
