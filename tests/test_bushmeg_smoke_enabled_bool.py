from __future__ import annotations

import pytest

import neureptrace.bushmeg_all_protocols as all_protocols
from neureptrace.bushmeg_all_protocols import method_registry


def _base_method_config() -> dict[str, object]:
    return {
        "participants": {"ids": "1,2,3,4"},
        "preprocessing": {"window_centers": [0.088, 0.136, 0.184]},
    }


def test_method_config_treats_quoted_false_smoke_enabled_as_false() -> None:
    spec = method_registry()["source_loso_logistic"]
    runner_config = {
        "method_settings": {
            spec.method: {
                "smoke_enabled": "false",
                "smoke_overrides": {
                    "participants": {"ids": "9"},
                    "preprocessing": {"window_centers": [9.0]},
                },
            }
        }
    }

    method_config = all_protocols._method_config(
        _base_method_config(),
        runner_config,
        spec,
        data_dir=None,
        participants=None,
        max_folds=1,
        include_heavy=False,
    )

    assert method_config["participants"]["ids"] == "1,2,3,4"
    assert method_config["preprocessing"]["window_centers"] == [0.088, 0.136, 0.184]


def test_method_config_applies_quoted_true_smoke_enabled() -> None:
    spec = method_registry()["source_loso_logistic"]
    runner_config = {
        "method_settings": {
            spec.method: {
                "smoke_enabled": "true",
                "smoke_overrides": {
                    "participants": {"ids": "9"},
                    "preprocessing": {"window_centers": [9.0]},
                },
            }
        }
    }

    method_config = all_protocols._method_config(
        _base_method_config(),
        runner_config,
        spec,
        data_dir=None,
        participants=None,
        max_folds=1,
        include_heavy=False,
    )

    assert method_config["participants"]["ids"] == "9"
    assert method_config["preprocessing"]["window_centers"] == [9.0]


def test_method_config_rejects_ambiguous_smoke_enabled_string() -> None:
    spec = method_registry()["source_loso_logistic"]

    with pytest.raises(ValueError, match="smoke_enabled"):
        all_protocols._method_config(
            _base_method_config(),
            {"method_settings": {spec.method: {"smoke_enabled": "maybe"}}},
            spec,
            data_dir=None,
            participants=None,
            max_folds=1,
            include_heavy=False,
        )


def test_method_settings_normalize_quoted_enabled_and_heavy_booleans() -> None:
    spec = method_registry()["source_loso_logistic"]

    settings = all_protocols._method_settings(
        {"method_settings": {spec.method: {"enabled": "false", "heavy": "true", "smoke_enabled": "false"}}},
        spec.method,
    )

    assert settings["enabled"] is False
    assert settings["heavy"] is True
    assert settings["smoke_enabled"] is False


def test_method_availability_treats_quoted_false_enabled_as_disabled() -> None:
    spec = method_registry()["source_loso_logistic"]
    settings = all_protocols._method_settings(
        {"method_settings": {spec.method: {"enabled": "false"}}},
        spec.method,
    )

    available, reason = all_protocols._method_availability(
        spec,
        _base_method_config(),
        settings=settings,
        include_heavy=False,
        max_folds=None,
    )

    assert not available
    assert "disabled" in reason


def test_method_availability_treats_quoted_false_smoke_enabled_as_false_for_smoke_runs() -> None:
    spec = method_registry()["source_loso_logistic"]
    settings = all_protocols._method_settings(
        {"method_settings": {spec.method: {"enabled": "false", "smoke_enabled": "false"}}},
        spec.method,
    )

    available, reason = all_protocols._method_availability(
        spec,
        _base_method_config(),
        settings=settings,
        include_heavy=False,
        max_folds=1,
    )

    assert not available
    assert "disabled" in reason


def test_method_availability_allows_disabled_method_when_quoted_true_smoke_run_is_requested() -> None:
    spec = method_registry()["source_loso_logistic"]
    settings = all_protocols._method_settings(
        {"method_settings": {spec.method: {"enabled": "false", "smoke_enabled": "true"}}},
        spec.method,
    )

    available, reason = all_protocols._method_availability(
        spec,
        _base_method_config(),
        settings=settings,
        include_heavy=False,
        max_folds=1,
    )

    assert available
    assert reason == ""


def test_method_settings_rejects_ambiguous_enabled_string() -> None:
    spec = method_registry()["source_loso_logistic"]

    with pytest.raises(ValueError, match="enabled"):
        all_protocols._method_settings(
            {"method_settings": {spec.method: {"enabled": "maybe"}}},
            spec.method,
        )
