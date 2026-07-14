from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.source_gaussian import fit_source_gaussian_decoder, gaussian_log_likelihoods
from neureptrace.decoding.source_mahalanobis import fit_source_mahalanobis_decoder, mahalanobis_distances, tied_covariance

SourceDecoder = Callable[..., object]


@pytest.mark.parametrize("decoder", [fit_source_gaussian_decoder, fit_source_mahalanobis_decoder])
@pytest.mark.parametrize("field", ["source_features", "test_features"])
def test_source_decoders_reject_boolean_feature_matrices(decoder: SourceDecoder, field: str) -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [2.0, 2.0],
            [2.2, 2.1],
        ],
        dtype=float,
    )
    test_features = np.asarray([[0.1, 0.0], [2.1, 2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    kwargs = {
        "source_features": source_features,
        "source_labels": source_labels,
        "test_features": test_features,
    }
    if field == "source_features":
        kwargs[field] = np.asarray(
            [
                [True, False],
                [False, True],
                [True, True],
                [False, False],
            ],
            dtype=bool,
        )
    else:
        kwargs[field] = np.asarray([[True, False], [False, True]], dtype=bool)

    with pytest.raises(ValueError, match=f"{field}.*boolean flags"):
        decoder(**kwargs)


@pytest.mark.parametrize("decoder", [fit_source_gaussian_decoder, fit_source_mahalanobis_decoder])
def test_source_decoders_reject_mixed_boolean_feature_values(decoder: SourceDecoder) -> None:
    with pytest.raises(ValueError, match="test_features.*boolean flags"):
        decoder(
            source_features=[[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.2, 2.1]],
            source_labels=["left", "left", "right", "right"],
            test_features=[[True, 0.0], [False, 1.0]],
        )


def test_gaussian_log_likelihoods_rejects_boolean_feature_inputs() -> None:
    with pytest.raises(ValueError, match="features.*boolean flags"):
        gaussian_log_likelihoods(
            np.asarray([[True, False]], dtype=bool),
            means=[[0.0, 0.0], [1.0, 1.0]],
            variances=[[1.0, 1.0], [1.0, 1.0]],
        )

    with pytest.raises(ValueError, match="means.*boolean flags"):
        gaussian_log_likelihoods(
            [[0.0, 0.0]],
            means=np.asarray([[True, False], [False, True]], dtype=bool),
            variances=[[1.0, 1.0], [1.0, 1.0]],
        )


def test_mahalanobis_helpers_reject_boolean_feature_inputs() -> None:
    with pytest.raises(ValueError, match="features.*boolean flags"):
        tied_covariance(
            np.asarray([[True, False], [False, True]], dtype=bool),
            labels=["left", "right"],
        )

    with pytest.raises(ValueError, match="features.*boolean flags"):
        mahalanobis_distances(
            np.asarray([[True, False]], dtype=bool),
            means=[[0.0, 0.0], [1.0, 1.0]],
            precision=np.eye(2),
        )
