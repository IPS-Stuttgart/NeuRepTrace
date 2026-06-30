def test_source_confidence_weighting_config_validation_smoke():
    from neureptrace.decoding.source_confidence_weighting import source_confidence_weight_config

    assert source_confidence_weight_config(normalize_weights="false").normalize_weights is False
