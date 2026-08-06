from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _noop(*args, **kwargs):
    del args, kwargs
    return None


def _load_benchmark_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    package = _module("neureptrace", __path__=[])
    decoding = _module("neureptrace.decoding", __path__=[])
    support = _module(
        "neureptrace._katja_finger_sequence_support",
        DEFAULT_CALIBRATION_COUNTS=(1, 3),
        DEFAULT_CALIBRATION_SEEDS=(13, 29),
        DEFAULT_PARTICIPANTS=("s05", "s06"),
        JULIA_FULL_FINETUNE_ACCURACY={},
        _composite_trial_ids=_noop,
        _constant_trial_values=_noop,
        _fit_source_preprocessor=_noop,
        _load_source_map=_noop,
        _mean_sem=_noop,
        _parse_csv_values=_noop,
        _stable_source_selection=_noop,
        _transform_features=_noop,
        derive_participant_local_finger_labels=_noop,
        load_katja_feature_cache=_noop,
    )
    progressive = _module(
        "neureptrace.decoding.progressive_sequence_finetune",
        TorchProgressiveSequenceClassifier=type(
            "TorchProgressiveSequenceClassifier",
            (),
            {},
        ),
        pack_complete_trial_events=_noop,
        permutation_constrained_decode=_noop,
        select_nested_trial_calibration_splits=_noop,
    )
    for name, module in {
        "neureptrace": package,
        "neureptrace.decoding": decoding,
        "neureptrace._katja_finger_sequence_support": support,
        "neureptrace.decoding.progressive_sequence_finetune": progressive,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    script = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "neureptrace"
        / "katja_finger_sequence_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_katja_finger_sequence_benchmark",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"calibration_counts": ()},
            "calibration_counts must not be empty",
            id="empty-counts",
        ),
        pytest.param(
            {"calibration_counts": (1, 1.0)},
            "calibration_counts must contain unique values",
            id="duplicate-counts-after-normalization",
        ),
        pytest.param(
            {"calibration_counts": (1, 2.5)},
            "calibration_counts value must be an integer",
            id="fractional-count",
        ),
        pytest.param(
            {"calibration_seeds": (13, 13.0)},
            "calibration_seeds must contain unique values",
            id="duplicate-seeds-after-normalization",
        ),
        pytest.param(
            {"calibration_seeds": (True,)},
            "calibration_seeds value must be an integer",
            id="boolean-seed",
        ),
        pytest.param(
            {"event_positions": (2, 3, 4, 4.0)},
            "event_positions must contain unique values",
            id="duplicate-event-position",
        ),
        pytest.param(
            {"event_positions": (2, 3, 4, 5.5)},
            "event_positions value must be an integer",
            id="fractional-event-position",
        ),
        pytest.param(
            {"n_source_participants": 9.5},
            "n_source_participants must be an integer",
            id="fractional-source-count",
        ),
        pytest.param(
            {"source_selection_seed": True},
            "source_selection_seed must be an integer",
            id="boolean-source-seed",
        ),
        pytest.param(
            {"pca_components": 64.5},
            "pca_components must be an integer",
            id="fractional-pca-components",
        ),
    ],
)
def test_benchmark_rejects_lossy_controls_before_cache_access(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
) -> None:
    module = _load_benchmark_module(monkeypatch)

    with pytest.raises(ValueError, match=message):
        module.run_katja_finger_sequence_benchmark({}, **kwargs)


def test_benchmark_accepts_exact_integral_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_benchmark_module(monkeypatch)

    with pytest.raises(KeyError, match="subjects"):
        module.run_katja_finger_sequence_benchmark(
            {},
            calibration_counts=(1.0, "3"),
            calibration_seeds=(13.0, "29"),
            event_positions=(2.0, "3", 4, 5),
            n_source_participants=1.0,
            source_selection_seed="13",
            pca_components=2.0,
        )
