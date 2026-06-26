# Subspace adaptation

`neureptrace.decoding.subspace_adaptation` implements Category-2 TCA-style source-target latent projection.

The fit uses source features and unlabeled target features. Optional source labels can be used to balance the source side of the marginal domain discrepancy. Held-out target labels are not part of the public API.

::: neureptrace.decoding.subspace_adaptation
    options:
      members:
        - SubspaceAdaptationConfig
        - SubspaceAdaptationResult
        - fit_subspace_adaptation
        - transform_subspace_features
        - subspace_adaptation_config
        - normalize_subspace_method
