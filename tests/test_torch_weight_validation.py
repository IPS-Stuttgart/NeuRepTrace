from __future__ import annotations

import pytest

from neureptrace.decoding.cdan import TorchCDANClassifier
from neureptrace.decoding.dann import TorchDANNClassifier
from neureptrace.decoding.source_domain_generalization import TorchSourceDomainGeneralizationClassifier
from neureptrace.decoding.source_vrex import TorchVRExClassifier


SOURCE_FEATURES = [[0.0], [1.0], [2.0], [3.0]]
SOURCE_LABELS = [0, 0, 1, 1]
SOURCE_DOMAINS = ["s1", "s2", "s1", "s2"]
TARGET_FEATURES = [[0.5], [2.5]]


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
