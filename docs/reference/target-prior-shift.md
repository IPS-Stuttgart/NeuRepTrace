# Target prior-shift adaptation

`neureptrace.decoding.target_prior_shift` adapts source-model probability rows to an unlabeled target batch by estimating target class priors with an EM prior-shift update.

The protocol is **Category 2 / unlabeled target-adaptive**. It uses target probability rows from a source-trained model and optional source priors, but it does not accept held-out target labels.

Typical usage:

```python
from neureptrace.decoding.target_prior_shift import adapt_target_probabilities_prior_shift

result = adapt_target_probabilities_prior_shift(
    probabilities=target_probabilities,
    source_prior=source_class_prior,
)

adapted_probabilities = result.probabilities
estimated_target_prior = result.estimated_target_prior
```

::: neureptrace.decoding.target_prior_shift
    options:
      members:
        - TargetPriorShiftConfig
        - TargetPriorShiftResult
        - adapt_target_probabilities_prior_shift
        - target_prior_shift_config
        - normalize_initial_prior
