from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.decoding import source_mixup


def _load_core_source_mixup_module():
    """Load the implementation file directly, bypassing package-level compatibility patches."""

    module_path = Path(source_mixup.__file__)
    spec = importlib.util.spec_from_file_location("_neureptrace_source_mixup_core_seed_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_source_mixup_config_rejects_negative_random_state() -> None:
    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        source_mixup.source_mixup_config(random_state=-1)


def test_source_mixup_config_accepts_none_random_state_strings() -> None:
    for value in ["none", "null", np.asarray("none")]:
        assert source_mixup.source_mixup_config(random_state=value).random_state is None


def test_source_mixup_config_accepts_scalar_array_random_state() -> None:
    assert source_mixup.source_mixup_config(random_state=np.asarray(7)).random_state == 7


def test_source_mixup_core_config_accepts_scalar_array_random_state_without_runtime_patch() -> None:
    core_module = _load_core_source_mixup_module()

    assert core_module.source_mixup_config(random_state=np.asarray("none")).random_state is None
    assert core_module.source_mixup_config(random_state=np.asarray("NULL")).random_state is None
    assert core_module.source_mixup_config(random_state=np.asarray(7)).random_state == 7


def test_source_mixup_core_config_rejects_container_random_state_without_runtime_patch() -> None:
    core_module = _load_core_source_mixup_module()

    for value in [[1], np.asarray([1])]:
        with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
            core_module.source_mixup_config(random_state=value)


def test_source_mixup_dataclass_config_rejects_negative_random_state_before_rng() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["class_a", "class_a", "class_b", "class_b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s1", "s2"], dtype=object)
    cfg = source_mixup.SourceMixUpConfig(synthetic_per_class=1, random_state=-1)

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        source_mixup.augment_source_with_mixup(
            features,
            labels,
            source_domains=domains,
            config=cfg,
        )


def test_source_mixup_core_dataclass_config_normalizes_seed_before_rng_without_runtime_patch() -> None:
    core_module = _load_core_source_mixup_module()
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["class_a", "class_a", "class_b", "class_b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s1", "s2"], dtype=object)
    cfg = core_module.SourceMixUpConfig(synthetic_per_class=1, random_state=np.asarray("none"), preserve_original=False)

    result = core_module.augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config=cfg,
    )

    assert result.features.shape[0] == 2
    assert result.metadata["source_mixup_random_state"] == ""
