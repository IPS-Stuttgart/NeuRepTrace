import numpy as np
import pytest

from neureptrace.decoding.source_weighting import (
    SourceGroupWeightingConfig,
    dynamic_source_group_weights,
    source_group_weighting_config,
)


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


def test_source_group_weighting_config_accepts_direct_config_instances():
    cfg = SourceGroupWeightingConfig(
        mode="target",
        metric="Log-Loss",
        temperature=np.asarray(0.50),
        top_k=np.asarray(2),
        blend="0.25",
        hybrid_target_similarity_weight=np.asarray(0.75),
    )

    normalized = source_group_weighting_config(cfg)

    assert normalized.mode == "target_similarity"
    assert normalized.metric == "log_loss"
    assert normalized.temperature == pytest.approx(0.50)
    assert normalized.top_k == 2
    assert normalized.blend == pytest.approx(0.25)
    assert normalized.hybrid_target_similarity_weight == pytest.approx(0.75)


def test_source_group_weighting_config_overrides_direct_config_instances():
    cfg = SourceGroupWeightingConfig(
        mode="target",
        temperature=np.asarray(0.50),
        top_k=np.asarray(2),
        hybrid_target_similarity_weight=np.asarray(0.75),
    )

    normalized = source_group_weighting_config(
        cfg,
        mode="hybrid",
        temperature="0.2",
        top_k="none",
        hybrid_target_similarity_weight="0.25",
    )

    assert normalized.mode == "hybrid"
    assert normalized.temperature == pytest.approx(0.2)
    assert normalized.top_k is None
    assert normalized.hybrid_target_similarity_weight == pytest.approx(0.25)
