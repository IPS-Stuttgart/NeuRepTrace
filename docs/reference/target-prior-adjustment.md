# Target prior adjustment

`neureptrace.decoding.target_prior_adjustment` implements unlabeled target-prior probability adjustment.

The protocol is **Category 2 / unlabeled target-adaptive**. It uses probability rows from a source-trained model on held-out target rows, but it does not accept held-out target labels.

Supported estimators:

- `mean`
- `em`

::: neureptrace.decoding.target_prior_adjustment
    options:
      members:
        - TargetPriorAdjustmentConfig
        - TargetPriorAdjustmentResult
        - adjust_target_probabilities_to_prior
        - estimate_target_prior_mean
        - estimate_target_prior_em
        - target_prior_adjustment_config
        - normalize_prior_estimator
