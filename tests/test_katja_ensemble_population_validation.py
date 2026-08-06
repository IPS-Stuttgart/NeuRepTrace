from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _noop(*args, **kwargs):
    del args, kwargs
    return None


def _load_population_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    torch = _module(
        "torch",
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )
    source_sweep = _module(
        "katja_ensemble_source_sweep",
        _configurations=_noop,
        _member_specs=_noop,
        _model_kwargs=_noop,
        _model_seed=_noop,
        _prepare_rows=_noop,
        _source_space_specs=_noop,
    )
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
        _mean_sem=_noop,
        _stable_source_selection=_noop,
        _transform_features=_noop,
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
        "torch": torch,
        "katja_ensemble_source_sweep": source_sweep,
        "neureptrace": package,
        "neureptrace.decoding": decoding,
        "neureptrace._katja_finger_sequence_support": support,
        "neureptrace.decoding.progressive_sequence_finetune": progressive,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "katja_ensemble_population.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_katja_ensemble_population",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_population_cli_rejects_duplicate_seed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_population_module(monkeypatch)

    with pytest.raises(ValueError, match="unique values"):
        module._parse_int_csv("13,13,29")


def test_population_runner_rejects_duplicate_seeds_before_loading_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_population_module(monkeypatch)

    with pytest.raises(ValueError, match="calibration_seeds must be unique"):
        module.run_targets(
            {},
            targets=("s06",),
            calibration_counts=(1,),
            calibration_seeds=(13, 13),
        )


def test_population_summary_rejects_duplicate_fold_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_population_module(monkeypatch)
    duplicate_rows = pd.DataFrame(
        [
            {
                "configuration": "nine_single_protocol",
                "target": "s06",
                "k": 20,
                "seed": 13,
            },
            {
                "configuration": "nine_single_protocol",
                "target": "s06",
                "k": 20,
                "seed": 13,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate configuration/target/k/seed rows",
    ):
        module._summarize_population(duplicate_rows)
