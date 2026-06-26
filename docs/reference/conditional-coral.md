# Conditional CORAL

`neureptrace.decoding.conditional_coral` provides class-conditional CORAL alignment using source labels and classifier-generated target pseudo-labels or target probabilities.

The protocol is Category 2 / unlabeled target-adaptive. Target features are used for adaptation, but true target labels are not part of the public API.

::: neureptrace.decoding.conditional_coral
    options:
      members:
        - ConditionalCoralConfig
        - ConditionalCoralClassStats
        - ConditionalCoralResult
        - fit_conditional_coral_alignment
        - conditional_coral_config
        - normalize_conditional_coral_fallback
