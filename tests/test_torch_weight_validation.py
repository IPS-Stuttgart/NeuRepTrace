from __future__ import annotations

import numpy as np
import pytest

from neureptrace import _torch_weight_validation_patch as torch_weight_patch
from neureptrace._torch_weight_validation_patch import _small_stratified_holdout
from neureptrace.decoding import TorchMLPClassifier
from neureptrace.decoding.cdan import TorchCDANClassifier
from neureptrace.decoding.dann import TorchDANNClassifier
from neureptrace.decoding.source_domain_generalization import TorchSourceDomainGeneralizationClassifier
from neureptrace.decoding.source_vrex import TorchVRExClassifier


SOURCE_FEATURES = [[0.0], [1.0], [2.0], [3.0]]
SOURCE_LABELS = [0, 0, 1, 1]
SOURCE_DOMAINS = ["s1", "s2", "s1", "s2"]
TARGET_FEATURES = [[0.5], [2.5]]


def _many_class_source_target() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(16), 2)
    source_features = np.column_stack(
        [
            labels.astype(float),
            np.tile([0.0, 1.0], 16),
        ]
    )
    target_features = source_features[:5] + np.asarray([0.05, 0.0])
    return source_features.astype(float), labels, target_features.astype(float)


def test_torch_patch_install_skips_missing_optional_torch_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = torch_weight_patch.importlib.import_module
    attempted_modules: list[str] = []

    def import_module(name: str, *args, **kwargs):
        attempted_modules.append(name)
        if name == "neureptrace.decoding.cdan":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(torch_weight_patch.importlib, "import_module", import_module)

    torch_weight_patch.install()

    assert "neureptrace.decoding.cdan" in attempted_modules


def test_torch_patch_install_preserves_internal_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = torch_weight_patch.importlib.import_module

    def import_module(name: str, *args, **kwargs):
        if name == "neureptrace.decoding.cdan":
            raise ModuleNotFoundError("No module named 'neureptrace.decoding.cdan'", name="neureptrace.decoding.cdan")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(torch_weight_patch.importlib, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError, match="neureptrace.decoding.cdan"):
        torch_weight_patch.install()


def test_torch_mlp_rejects_unknown_class_weight_before_torch_initialization() -> None:
    model = TorchMLPClassifier(class_weight="balance")

    with pytest.raises(ValueError, match="class_weight must be None or 'balanced'"):
        model.fit(SOURCE_FEATURES, SOURCE_LABELS)


def test_dann_rejects_unknown_class_weight_before_torch_initialization() -> None:
    model = TorchDANNClassifier(class_weight="balance")

    with pytest.raises(ValueError, match="class_weight must be None or 'balanced'"):
        model.fit(SOURCE_FEATURES, SOURCE_LABELS, target_features=TARGET_FEATURES)


def test_cdan_rejects_unknown_class_weight_before_torch_initialization() -> None:
    model = TorchCDANClassifier(class_weight="balance")

    with pytest.raises(ValueError, match="class_weight must be None or 'balanced'"):
        model.fit(SOURCE_FEATURES, SOURCE_LABELS, target_features=TARGET_FEATURES)


def test_source_domain_generalization_rejects_unknown_class_weight() -> None:
    model = TorchSourceDomainGeneralizationClassifier(class_weight="balance")

    with pytest.raises(ValueError, match="class_weight must be None or 'balanced'"):
        model.fit(SOURCE_FEATURES, SOURCE_LABELS, source_domains=SOURCE_DOMAINS)


def test_source_domain_generalization_rejects_unknown_domain_weight() -> None:
    model = TorchSourceDomainGeneralizationClassifier(domain_weight="balance")

    with pytest.raises(ValueError, match="domain_weight must be None or 'balanced'"):
        model.fit(SOURCE_FEATURES, SOURCE_LABELS, source_domains=SOURCE_DOMAINS)


def test_source_vrex_rejects_unknown_class_weight() -> None:
    model = TorchVRExClassifier(class_weight="balance")

    with pytest.raises(ValueError, match="class_weight must be None or 'balanced'"):
        model.fit(SOURCE_FEATURES, SOURCE_LABELS, source_domains=SOURCE_DOMAINS)


def test_torch_mlp_validation_guard_preserves_composite_label_rows() -> None:
    labels = [
        ("cue", "left"),
        ("cue", "left"),
        ("cue", "right"),
        ("cue", "right"),
    ]

    assert _small_stratified_holdout(labels, 0.5) is False


def test_small_stratified_holdout_detects_validation_set_smaller_than_class_count() -> None:
    labels = np.repeat(np.arange(16), 2)

    assert _small_stratified_holdout(labels, 0.1)
    assert not _small_stratified_holdout(labels, 0.5)


def test_dann_restores_requested_fraction_after_small_validation_fallback() -> None:
    pytest.importorskip("torch")
    source_features, source_labels, target_features = _many_class_source_target()
    model = TorchDANNClassifier(
        hidden_units=8,
        embedding_dim=4,
        max_epochs=1,
        batch_size=8,
        patience=1,
        validation_fraction=0.1,
        random_state=7,
        device="cpu",
    )

    model.fit(source_features, source_labels, target_features=target_features)

    assert model.validation_fraction == 0.1
    assert model.n_classes_ == 16
    assert model.source_rows_ == 32


def test_cdan_restores_requested_fraction_after_small_validation_fallback() -> None:
    pytest.importorskip("torch")
    source_features, source_labels, target_features = _many_class_source_target()
    model = TorchCDANClassifier(
        hidden_units=8,
        embedding_dim=4,
        max_epochs=1,
        batch_size=8,
        patience=1,
        validation_fraction=0.1,
        random_state=7,
        device="cpu",
    )

    model.fit(source_features, source_labels, target_features=target_features)

    assert model.validation_fraction == 0.1
    assert model.n_classes_ == 16
    assert model.source_rows_ == 32


def test_torch_mlp_rejects_bad_prediction_feature_shape() -> None:
    pytest.importorskip("torch")
    model = TorchMLPClassifier(
        hidden_units=4,
        max_iter=1,
        batch_size=2,
        patience=1,
        validation_fraction=0.0,
        random_state=7,
    )
    model.fit(SOURCE_FEATURES, SOURCE_LABELS)

    with pytest.raises(ValueError, match="TorchMLPClassifier prediction features must be a two-dimensional"):
        model.predict_proba(np.asarray([0.5], dtype=float))
    with pytest.raises(ValueError, match="TorchMLPClassifier prediction features must be finite"):
        model.predict_proba(np.asarray([[np.nan]], dtype=float))
    with pytest.raises(ValueError, match="TorchMLPClassifier prediction features width 2 does not match fitted width 1"):
        model.predict(np.zeros((2, 2), dtype=float))


def test_dann_rejects_bad_prediction_feature_shape() -> None:
    pytest.importorskip("torch")
    model = TorchDANNClassifier(
        hidden_units=4,
        embedding_dim=2,
        max_epochs=1,
        batch_size=2,
        patience=1,
        validation_fraction=0.0,
        random_state=7,
        device="cpu",
    )
    model.fit(SOURCE_FEATURES, SOURCE_LABELS, target_features=TARGET_FEATURES)

    with pytest.raises(ValueError, match="TorchDANNClassifier prediction features must be a two-dimensional"):
        model.predict_proba(np.asarray([0.5], dtype=float))
    with pytest.raises(ValueError, match="TorchDANNClassifier prediction features must be finite"):
        model.predict_proba(np.asarray([[np.inf]], dtype=float))
    with pytest.raises(ValueError, match="TorchDANNClassifier prediction features width 2 does not match fitted width 1"):
        model.decision_function(np.zeros((2, 2), dtype=float))
