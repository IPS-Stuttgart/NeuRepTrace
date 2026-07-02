from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_knn import SourceKNNConfig, source_knn_config
from neureptrace.decoding.source_pca import SourcePCAConfig, source_pca_config
from neureptrace.decoding.source_polynomial import (
    SourcePolynomialConfig,
    fit_source_polynomial_reference,
    source_polynomial_config,
)


def test_source_pca_accepts_numpy_string_component_aliases() -> None:
    assert SourcePCAConfig(n_components=np.asarray("FULL")).n_components == "full"
    assert source_pca_config(n_components=np.asarray("all")).n_components == "all"


def test_source_knn_accepts_numpy_string_k_aliases() -> None:
    assert SourceKNNConfig(k=np.asarray("FULL")).k == "full"
    assert source_knn_config(k=np.asarray("all")).k == "all"


def test_source_polynomial_accepts_numpy_string_max_interaction_aliases() -> None:
    direct = SourcePolynomialConfig(max_interactions=np.asarray("full"))
    helper = source_polynomial_config(max_interactions=np.asarray("ALL"))

    assert direct.max_interactions == "all"
    assert helper.max_interactions == "all"

    reference = fit_source_polynomial_reference(3, config=direct)
    assert len(reference.interaction_pairs) == 3
