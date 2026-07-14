from __future__ import annotations

from neureptrace.features.oscillatory import BandFeatureWindow, _normalize_windows


def test_normalize_windows_preserves_single_window_generator() -> None:
    windows = (value for value in (-0.25, 0.25))

    normalized = _normalize_windows(windows)

    assert normalized == (BandFeatureWindow("window", -0.25, 0.25),)


def test_normalize_windows_preserves_generator_collection() -> None:
    bounds = ((-0.4, -0.1), (0.1, 0.4))
    windows = ((value for value in window) for window in bounds)

    normalized = _normalize_windows(windows)

    assert normalized == (
        BandFeatureWindow("window_0", -0.4, -0.1),
        BandFeatureWindow("window_1", 0.1, 0.4),
    )
