from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_centroid import SourceCentroidConfig, fit_source_centroid_decoder, source_centroid_config


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"temperature": np.asarray(1.0)}, "temperature"),
        ({"temperature": np.asarray([1.0])}, "temperature"),
        ({"shrinkage": np.asarray(0.25)}, "shrinkage"),
        ({"shrinkage": np.asarray([0.25])}, "shrinkage"),
        ({"epsilon": np.asarray(1e-8)}, "epsilon"),
        ({"epsilon": np.asarray([1e-8])}, "epsilon"),
    ],
)
def test_source_centroid_config_rejects_array_numeric_controls(kwargs, field) -> None:
    with pytest.raises(ValueError, match=field):
        source_centroid_config(**kwargs)


def test_source_centroid_config_preserves_numpy_numeric_scalars() -> None:
    cfg = source_centroid_config(
        temperature=np.float64(2.0),
        shrinkage=np.float64(0.25),
        epsilon=np.float32(1e-5),
    )

    assert cfg.temperature == 2.0
    assert cfg.shrinkage == 0.25
    assert cfg.epsilon == pytest.approx(1e-5)


def test_source_centroid_revalidates_direct_config_instances() -> None:
    source_features = np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-0.75], [0.75]], dtype=float)

    with pytest.raises(ValueError, match="temperature"):
        fit_source_centroid_decoder(
            source_features=source_features,
            source_labels=source_labels,
            test_features=test_features,
            config=SourceCentroidConfig(temperature=np.asarray([1.0])),
        )
