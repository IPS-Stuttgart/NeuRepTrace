from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import neureptrace.decoding.source_domain_mask as source_domain_mask_module
from neureptrace.decoding.source_domain_mask import source_domain_mask


def _load_core_source_domain_mask_module():
    """Load the implementation file directly, bypassing package-level compatibility patches."""

    module_path = Path(source_domain_mask_module.__file__)
    spec = importlib.util.spec_from_file_location("_neureptrace_source_domain_mask_core_seed_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_source_domain_mask_accepts_none_like_seed_values() -> None:
    for seed in [None, "", " none ", "NULL", np.asarray("none")]:
        result = source_domain_mask(["a", "a", "b", "b", "c", "c"], random_state=seed)
        assert result.metadata["source_domain_mask_random_state"] == ""


def test_source_domain_mask_accepts_scalar_array_seed_values() -> None:
    result = source_domain_mask(["a", "a", "b", "b", "c", "c"], random_state=np.asarray(7))
    assert result.metadata["source_domain_mask_random_state"] == 7


def test_source_domain_mask_core_accepts_scalar_array_seed_values_without_runtime_patch() -> None:
    core_module = _load_core_source_domain_mask_module()

    none_result = core_module.source_domain_mask(["a", "a", "b", "b", "c", "c"], random_state=np.asarray("none"))
    seeded_result = core_module.source_domain_mask(["a", "a", "b", "b", "c", "c"], random_state=np.asarray(7))

    assert none_result.metadata["source_domain_mask_random_state"] == ""
    assert seeded_result.metadata["source_domain_mask_random_state"] == 7


def test_source_domain_mask_rejects_invalid_seed_values() -> None:
    for seed in [True, -1, 1.5, [1], np.asarray([1])]:
        with pytest.raises(ValueError, match="random_state"):
            source_domain_mask(["a", "a", "b", "b"], random_state=seed)


def test_source_domain_mask_core_rejects_non_scalar_array_seeds_without_runtime_patch() -> None:
    core_module = _load_core_source_domain_mask_module()

    for seed in [[1], np.asarray([1])]:
        with pytest.raises(ValueError, match="random_state"):
            core_module.source_domain_mask(["a", "a", "b", "b"], random_state=seed)
