# Source bagging decoder

`neureptrace.decoding.source_bagging` implements a strict source-only bootstrap ensemble for feature decoding.

Each base estimator is trained on a source-only row sample and optional feature subset. Held-out rows are only scored, and probabilities are averaged across estimators.

This is **Category 1 / strict source-only**.

::: neureptrace.decoding.source_bagging
    options:
      members:
        - SourceBaggingConfig
        - SourceBaggingResult
        - fit_source_bagging_decoder
        - source_bagging_config
