# VREx domain generalization

`neureptrace.decoding.vrex` provides a strict source-only linear classifier that penalizes variance in cross-entropy risk across source domains.

The estimator uses source features, source labels, and source-domain identifiers. It does not use held-out target features or labels.

::: neureptrace.decoding.vrex
