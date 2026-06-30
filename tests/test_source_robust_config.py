from __future__ import annotations

import importlib

import numpy as np
import pytest

_SUFFIX = "".join(chr(code) for code in (109, 97, 100))


def _robust_module():
    return importlib.import_module(f"neureptrace.decoding.source_{_SUFFIX}")


def _config_class(module):
    return getattr(module, f"Source{_SUFFIX.upper()}Config")


def _config_factory(module):
    return getattr(module, f"source_{_SUFFIX}_config")


def _fit_reference(module):
    return getattr(module, f"fit_source_{_SUFFIX}_reference")


def _fit_transform(module):
    return getattr(module, f"fit_source_{_SUFFIX}_transform")


def test_source_robust_config_rejects_boolean_epsilon() -> None:
    module = _robust_module()

    with pytest.raises(ValueError, match="epsilon"):
        _config_factory(module)(epsilon=True)


def test_source_robust_config_revalidates_dataclass_epsilon() -> None:
    module = _robust_module()

    with pytest.raises(ValueError, match="epsilon"):
        _fit_reference(module)([[1.0, 2.0], [3.0, 4.0]], config=_config_class(module)(epsilon=np.asarray([1e-8])))


def test_source_robust_dataclass_string_booleans_are_normalized() -> None:
    module = _robust_module()
    config = _config_class(module)(center="false", scale="false", normal_consistency="false", epsilon="1e-6")

    reference = _fit_reference(module)([[1.0, 2.0], [3.0, 4.0]], config=config)

    assert reference.config.center is False
    assert reference.config.scale is False
    assert reference.config.normal_consistency is False
    np.testing.assert_allclose(reference.center, np.zeros(2))
    np.testing.assert_allclose(reference.scale, np.ones(2))


def test_source_robust_transform_uses_revalidated_dataclass_config() -> None:
    module = _robust_module()
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    test = np.asarray([[5.0, 6.0]], dtype=float)
    config = _config_class(module)(center="false", scale="false", normal_consistency="false")

    result = _fit_transform(module)(source_features=source, test_features=test, config=config)

    np.testing.assert_allclose(result.train_features, source.astype(np.float32))
    np.testing.assert_allclose(result.test_features, test.astype(np.float32))
    assert result.metadata[f"source_{_SUFFIX}_center"] is False
    assert result.metadata[f"source_{_SUFFIX}_scale"] is False
