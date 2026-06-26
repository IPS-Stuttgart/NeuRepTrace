# Sampled geodesic flow

`neureptrace.decoding.geodesic_flow` implements dependency-light sampled geodesic-flow features for cross-subject transfer.

The protocol is **Category 2 / unlabeled target-adaptive**. It fits a PCA basis on source features and a second PCA basis on unlabeled target features, samples orthonormal intermediate bases between them, and represents rows by concatenating projections onto the sampled bases.

No target labels are accepted by the public API.

Typical usage:

```python
from neureptrace.decoding.geodesic_flow import fit_sampled_geodesic_flow_features

result = fit_sampled_geodesic_flow_features(
    source_features=X_source,
    target_adaptation_features=X_target_calib,  # optional; otherwise target_test is transductive
    target_test_features=X_target_test,
    config={"n_components": 16, "n_steps": 5},
)

X_source_gf = result.train_features
X_target_gf = result.test_features
```

::: neureptrace.decoding.geodesic_flow
    options:
      members:
        - GeodesicFlowConfig
        - GeodesicFlowBasis
        - GeodesicFlowResult
        - fit_sampled_geodesic_flow_features
        - geodesic_flow_config
        - sample_geodesic_bases
        - transform_with_geodesic_bases
