from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_full_curve_module():
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    script_path = experiments / "katja_ensemble_full_curve.py"
    population_stub = types.ModuleType("katja_ensemble_population")
    population_stub.PRIMARY_CONFIGURATION = "hybrid_source_model_ensemble6"
    population_stub._summarize_population = lambda _frame: None
    previous = sys.modules.get("katja_ensemble_population")
    sys.modules["katja_ensemble_population"] = population_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "test_katja_ensemble_full_curve_module",
            script_path,
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("katja_ensemble_population", None)
        else:
            sys.modules["katja_ensemble_population"] = previous
    return module


def test_locked_full_curve_rejects_incomplete_target_cohort(tmp_path: Path):
    full_curve = _load_full_curve_module()
    feature_cache = tmp_path / "katja-cache.npz"
    feature_cache.touch()

    with pytest.raises(
        ValueError,
        match=r"exact target cohort; missing=\['s28'\], unexpected=\[\]",
    ):
        full_curve.run_full_curve(
            feature_cache=feature_cache,
            output_root=tmp_path / "output",
            targets=full_curve.DEFAULT_TARGETS[:-1],
        )
