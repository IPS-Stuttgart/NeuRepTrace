from __future__ import annotations

import pytest

from neureptrace.decoding import TorchMLPClassifier


SOURCE_FEATURES = [[0.0], [1.0], [2.0], [3.0]]
SOURCE_LABELS = [0, 0, 1, 1]


def test_torch_weight_validation_patch_installed_by_package_import(monkeypatch: pytest.MonkeyPatch) -> None:
    assert getattr(
        TorchMLPClassifier.fit,
        "_neureptrace_torch_weight_validation_patch_installed",
        False,
    )

    model = TorchMLPClassifier(hidden_units=True)

    def fail_torch_import():
        raise AssertionError("torch should not be initialized before scalar option validation")

    monkeypatch.setattr(model, "_torch", fail_torch_import)

    with pytest.raises(ValueError, match="hidden_units must be a positive integer"):
        model.fit(SOURCE_FEATURES, SOURCE_LABELS)
