import numpy as np
import pytest

from neureptrace.decoding.source_vrex import TorchVRExClassifier


def test_vrex_metadata():
    metadata = TorchVRExClassifier().metadata()
    assert metadata["source_vrex_protocol"] == "source_only_vrex"
    assert metadata["source_vrex_valid_for_strict_source_only"] is True


def test_vrex_fit_rejects_nonfinite_source_features_before_torch_training():
    source_features = np.asarray(
        [
            [0.0, 0.1],
            [0.2, np.nan],
            [1.0, 1.1],
            [1.2, 1.3],
        ],
        dtype=float,
    )
    source_labels = np.asarray([0, 0, 1, 1])
    source_domains = np.asarray(["subject-a", "subject-b", "subject-a", "subject-b"], dtype=object)

    with pytest.raises(ValueError, match="finite"):
        TorchVRExClassifier(max_epochs=1).fit(source_features, source_labels, source_domains=source_domains)
