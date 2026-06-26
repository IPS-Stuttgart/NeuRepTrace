# Joint distribution adaptation

`neureptrace.decoding.joint_distribution_adaptation` implements iterative Category-2 alignment of marginal and class-conditional source-target distributions.

The method uses source labels and unlabeled target features. Target class structure is represented by pseudo-labels or optional source-model target probabilities. Held-out target labels are not part of the API.

::: neureptrace.decoding.joint_distribution_adaptation
    options:
      members:
        - JointDistributionAdaptationConfig
        - JointDistributionAdaptationResult
        - fit_joint_distribution_adaptation
        - transform_joint_distribution_features
        - joint_distribution_adaptation_config
        - normalize_jda_method
