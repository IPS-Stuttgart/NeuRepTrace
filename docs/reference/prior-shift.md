# Unlabeled prior-shift adaptation

`neureptrace.decoding.prior_shift` adapts target probability traces when a source-trained decoder is evaluated on a held-out target batch with a different class prior.

The method is **Category 2 / unlabeled target-adaptive**. It uses target probability rows, and optionally a source training prior, but it does not accept held-out target labels.

Typical use:

```python
from neureptrace.decoding.prior_shift import adapt_probabilities_for_prior_shift

result = adapt_probabilities_for_prior_shift(
    target_probabilities,
    source_prior=[0.5, 0.5],
)
adapted_probabilities = result.probabilities
```

For run-wise or block-wise shifts, use `adapt_probability_blocks_for_prior_shift` with a block id per target row. The block-wise protocol is still target-label-free, but it should be reported separately from strict source-only decoding because the held-out target probability distribution is used for adaptation.

::: neureptrace.decoding.prior_shift
    options:
      members:
        - PriorShiftAdaptationResult
        - PriorShiftBlockResult
        - adapt_probabilities_for_prior_shift
        - adapt_probability_blocks_for_prior_shift
        - reweight_probabilities_by_prior
        - prior_from_labels
