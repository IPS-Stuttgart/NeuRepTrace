import numpy as np
import pytest

from neureptrace.decoding.transfer_component_analysis import (
    fit_tca_transfer_classifier,
    transfer_component_analysis_features,
)


def _tca_toy_data():
    source = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 1.0],
            [1.2, 1.0],
        ],
        dtype=float,
    )
    target = np.array(
        [
            [0.1, 0.1],
            [1.1, 1.1],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1])
    return source, target, labels


@pytest.mark.parametrize("gamma", [True, False, np.bool_(True)])
def test_tca_rejects_boolean_gamma(gamma):
    source, target, _labels = _tca_toy_data()

    with pytest.raises(ValueError, match="gamma must be positive"):
        transfer_component_analysis_features(
            source,
            target,
            kernel="rbf",
            gamma=gamma,
            n_components=1,
        )


@pytest.mark.parametrize("n_components", [True, False, np.bool_(True)])
def test_tca_rejects_boolean_component_counts(n_components):
    source, target, _labels = _tca_toy_data()

    with pytest.raises(ValueError, match="n_components"):
        transfer_component_analysis_features(
            source,
            target,
            n_components=n_components,
        )


def test_tca_classifier_rejects_boolean_sample_weight_masks():
    source, target, labels = _tca_toy_data()

    with pytest.raises(ValueError, match="sample_weight"):
        fit_tca_transfer_classifier(
            source_features=source,
            source_labels=labels,
            target_features=target,
            n_components=1,
            sample_weight=[True, False, True, False],
        )
