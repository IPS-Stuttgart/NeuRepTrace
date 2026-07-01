from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_bagging import SourceBaggingConfig, fit_source_bagging_decoder


def test_source_bagging_revalidates_direct_config_string_controls() -> None:
    source_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    source_labels = np.asarray(["class_a", "class_a", "class_b", "class_b"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config=SourceBaggingConfig(
            n_estimators="1",
            sample_fraction="1",
            feature_fraction="1",
            bootstrap_rows="false",
            bootstrap_features="true",
            class_balanced="false",
            random_state="none",
            epsilon="1e-6",
        ),
    )

    assert result.n_estimators == 1
    assert result.metadata["source_bagging_sample_fraction"] == pytest.approx(1.0)
    assert result.metadata["source_bagging_feature_fraction"] == pytest.approx(1.0)
    assert result.metadata["source_bagging_bootstrap_rows"] is False
    assert result.metadata["source_bagging_bootstrap_features"] is True
    assert result.metadata["source_bagging_class_balanced"] is False
    assert result.metadata["source_bagging_random_state"] == ""
