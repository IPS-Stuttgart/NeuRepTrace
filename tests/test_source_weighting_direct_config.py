import numpy as np
import pytest

from neureptrace.decoding.source_weighting import SourceGroupWeightingConfig, dynamic_source_group_weights


def test_dynamic_source_group_weights_normalizes_direct_config_mode_aliases():
    cfg = SourceGroupWeightingConfig(mode="source-reliability", temperature=np.asarray(0.10), top_k=np.asarray(1))

    weights = dynamic_source_group_weights(config=cfg, source_scores={"best": 0.90, "weak": 0.20})

    assert weights["best"] > weights["weak"]
    assert weights["weak"] == pytest.approx(0.0)
    assert np.mean(list(weights.values())) == pytest.approx(1.0)


def test_dynamic_source_group_weights_normalizes_direct_disabled_config_aliases():
    cfg = SourceGroupWeightingConfig(mode="false")

    weights = dynamic_source_group_weights(config=cfg, groups=["s1", "s2"])

    assert weights == {"s1": 1.0, "s2": 1.0}
