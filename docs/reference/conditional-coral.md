# Conditional CORAL

`neureptrace.decoding.conditional_coral` implements a dependency-light class-wise CORAL transform for pseudo-class adaptation.

The module fits source class statistics and pseudo-class statistics in the adaptation set, then maps source rows into the adapted feature space. Sparse pseudo classes can fall back to a global CORAL map.

::: neureptrace.decoding.conditional_coral
    options:
      members:
        - ConditionalCoralConfig
        - CoralStats
        - ConditionalCoralResult
        - fit_pseudo_conditional_coral
        - conditional_coral_config
        - coral_transform
