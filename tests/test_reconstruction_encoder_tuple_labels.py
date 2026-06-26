from __future__ import annotations

import numpy as np

from neureptrace.decoding.reconstruction_encoder import (
    RECONSTRUCTION_SOURCE_ONLY,
    fit_reconstruction_latent_classifier,
    reconstruction_encoder_config,
)


def test_reconstruction_latent_classifier_preserves_tuple_labels() -> None:
    labels = np.repeat([0, 1], 12)
    prototypes = np.asarray([[2.0, 0.0, 0.3, 0.0], [0.0, 2.0, 0.0, 0.3]], dtype=float)
    source = prototypes[labels]
    target = source + np.asarray([0.2, -0.1, 0.0, 0.1], dtype=float)
    tuple_labels = [("motor", int(label)) for label in labels.tolist()]

    result = fit_reconstruction_latent_classifier(
        train_features=source,
        train_labels=tuple_labels,
        test_features=target,
        config=reconstruction_encoder_config(fit_scope=RECONSTRUCTION_SOURCE_ONLY, n_components=2),
    )

    assert result.predictions.shape == (target.shape[0],)
    assert result.probabilities is not None
    assert result.probabilities.shape == (target.shape[0], 2)
    assert result.classes.shape == (2,)
    assert result.classes.tolist() == [("motor", 0), ("motor", 1)]
    assert all(isinstance(label, tuple) for label in result.predictions.tolist())
    assert result.metadata["classifier_n_classes"] == 2
    assert result.metadata["classifier_label_encoding"] == "integer_codes_for_composite_labels"
