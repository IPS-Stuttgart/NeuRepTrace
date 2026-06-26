import pytest

pytest.importorskip("torch")

from neureptrace.decoding.source_vrex import TorchVRExClassifier


def test_vrex_metadata():
    metadata = TorchVRExClassifier().metadata()
    assert metadata["source_vrex_protocol"] == "source_only_vrex"
    assert metadata["source_vrex_valid_for_strict_source_only"] is True
