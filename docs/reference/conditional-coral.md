# Conditional CORAL

`neureptrace.decoding.conditional_coral` implements pseudo-label class-conditional CORAL for cross-subject transfer.

The protocol is **Category 2 / unlabeled target-adaptive**. It uses source features and source labels, plus unlabeled target features with pseudo-labels or probability predictions. It does not accept held-out target labels.

Supported pseudo-label sources:

- caller-supplied `target_pseudo_labels`,
- caller-supplied `target_probabilities` in source-class order,
- a source-trained classifier fitted internally when neither is supplied.

If a pseudo-class has too few confident target rows, the implementation can fall back to global target statistics or raise an error.

::: neureptrace.decoding.conditional_coral
    options:
      members:
        - ConditionalCoralConfig
        - CoralClassStats
        - ConditionalCoralResult
        - fit_pseudo_label_conditional_coral
        - conditional_coral_config
        - coral_align_features
        - feature_stats
        - normalize_conditional_coral_fallback
