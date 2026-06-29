from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from neureptrace.decoding import source_jitter as installed_source_jitter


def _load_direct_source_jitter_module() -> ModuleType:
    path = Path(installed_source_jitter.__file__)
    module_name = "_neureptrace_direct_source_jitter_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_direct_core_jitter_config_normalizes_scalar_array_options() -> None:
    source_jitter = _load_direct_source_jitter_module()

    cfg = source_jitter.source_feature_jitter_config(
        preserve_original=np.asarray(0),
        random_state=np.asarray(" NULL "),
    )

    assert cfg.preserve_original is False
    assert cfg.random_state is None


@pytest.mark.parametrize("value", [[1], {"seed": 1}, np.asarray([1, 2])])
def test_direct_core_jitter_config_rejects_container_random_state_as_value_error(value: object) -> None:
    source_jitter = _load_direct_source_jitter_module()

    with pytest.raises(ValueError, match="random_state"):
        source_jitter.source_feature_jitter_config(random_state=value)


@pytest.mark.parametrize("value", [np.asarray([1]), 2, "maybe"])
def test_direct_core_jitter_config_rejects_ambiguous_preserve_original_as_value_error(value: object) -> None:
    source_jitter = _load_direct_source_jitter_module()

    with pytest.raises(ValueError, match="preserve_original"):
        source_jitter.source_feature_jitter_config(preserve_original=value)
