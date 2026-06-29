# Target probability smoothing

`neureptrace.decoding.target_probability_smoothing` implements graph smoothing of source-model probability rows over unlabeled held-out feature geometry.

The protocol is **Category 2 / unlabeled target-adaptive**. It uses held-out target features to build a graph and smooth initial probability rows, but it does not accept held-out target labels.

Typical usage:

```python
from neureptrace.decoding.target_probability_smoothing import smooth_target_probabilities

result = smooth_target_probabilities(
    target_features=X_target,
    probabilities=source_model_probabilities,
    config={"alpha": 0.5, "n_neighbors": 8},
)

smoothed_probabilities = result.probabilities
```

::: neureptrace.decoding.target_probability_smoothing
    options:
      members:
        - TargetProbabilitySmoothingConfig
        - TargetProbabilitySmoothingResult
        - smooth_target_probabilities
        - target_probability_smoothing_config
        - rbf_affinity
        - row_normalize
