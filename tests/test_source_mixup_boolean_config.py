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
    spec = importlib.util.spec_from_file_location("_neureptrace_source_mixup_core_bool_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_source_mixup_config_normalizes_boolean_like_strings() -> None:
    cfg = source_mixup.source_mixup_config(
        same_class_partner="false",
        cross_domain_partner="0",
        preserve_original="off",
    )

    assert cfg.same_class_partner is False
    assert cfg.cross_domain_partner is False
    assert cfg.preserve_original is False


def test_source_mixup_core_config_normalizes_boolean_like_strings_without_runtime_patch() -> None:
    core_module = _load_core_source_mixup_module()

    cfg = core_module.source_mixup_config(
        same_class_partner="false",
        cross_domain_partner=np.asarray("0"),
        preserve_original="off",
    )

    assert cfg.same_class_partner is False
    assert cfg.cross_domain_partner is False
    assert cfg.preserve_original is False


def test_source_mixup_config_rejects_ambiguous_boolean_strings() -> None:
    with pytest.raises(ValueError, match="same_class_partner must be a boolean value"):
        source_mixup.source_mixup_config(same_class_partner="maybe")


def test_source_mixup_core_config_rejects_ambiguous_boolean_values_without_runtime_patch() -> None:
    core_module = _load_core_source_mixup_module()

    for value in ["maybe", 2, np.asarray([True])]:
        with pytest.raises(ValueError, match="same_class_partner must be a boolean value"):
            core_module.source_mixup_config(same_class_partner=value)


def test_source_mixup_mapping_config_can_disable_original_rows_with_string_false() -> None:
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

    result = source_mixup.augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config={
            "synthetic_per_class": 1,
            "same_class_partner": "true",
            "cross_domain_partner": "false",
            "preserve_original": "false",
            "random_state": 0,
        },
    )

    assert result.features.shape[0] == 2
    assert result.labels.shape == (2,)
    assert result.synthetic_mask.tolist() == [True, True]
    assert result.metadata["source_mixup_preserve_original"] is False


def test_source_mixup_core_mapping_config_disables_original_rows_with_string_false_without_runtime_patch() -> None:
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

    result = core_module.augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config={
            "synthetic_per_class": 1,
            "same_class_partner": "true",
            "cross_domain_partner": "false",
            "preserve_original": "false",
            "random_state": 0,
        },
    )

    assert result.features.shape[0] == 2
    assert result.labels.shape == (2,)
    assert result.synthetic_mask.tolist() == [True, True]
    assert result.metadata["source_mixup_preserve_original"] is False
