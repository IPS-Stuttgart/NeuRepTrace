# Source-domain probability ensemble

`neureptrace.decoding.source_ensemble` trains one classifier per source domain and combines held-out target probabilities.

Uniform weighting is Category 1 because target rows are only scored. Non-uniform weighting modes are Category 2 because unlabeled target predictions or target feature distributions affect the source-domain weights.

Supported weighting modes are `uniform`, `target_confidence`, `target_entropy`, and `target_feature_similarity`.

The public API intentionally has no `target_labels` argument.

::: neureptrace.decoding.source_ensemble
    options:
      members:
        - SourceDomainModel
        - SourceDomainEnsembleResult
        - fit_source_domain_probability_ensemble
        - normalize_ensemble_weighting
