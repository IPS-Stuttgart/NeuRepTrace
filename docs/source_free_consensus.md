# Source-free consensus adaptation

`neureptrace.decoding.source_free_consensus` adds a Protocol 2.5-style
consensus wrapper for OpenNeuro source-free target adaptation.

The motivation is that difficult LOSO folds can produce several plausible but
fragile source-free outputs:

- the raw source posterior may be stable but source-prior biased;
- balanced pseudo-label adaptation may recover minority classes but be noisy;
- robust target-prior correction may improve balanced accuracy but overcorrect on
  a small or genuinely imbalanced target subject.

The consensus wrapper runs several such variants and combines their posterior
probabilities without using held-out target labels.

## Minimal usage

```python
from neureptrace.decoding.source_free_consensus import (
    fit_source_free_consensus_predict_proba,
)

result = fit_source_free_consensus_predict_proba(
    source_model=source_model,
    target_features=X_target_unlabeled,
)

probabilities = result.probabilities
metadata = result.metadata
```

The default recipe combines:

- `source_raw`: frozen source posterior;
- `balanced_topk`: source-free prototype adaptation with balanced pseudo-label
  row selection;
- `robust_prior`: balanced pseudo-label adaptation plus robust target-prior
  correction.

## Unlabeled consensus weighting

If no fixed variant weights are supplied, the wrapper estimates target-batch
variant weights from two unlabeled diagnostics:

- row confidence: lower mean normalized entropy is better;
- marginal class balance: higher entropy of the predicted class marginal is
  better.

This intentionally avoids simply selecting the most confident variant, because a
source model can be confidently collapsed to one pseudo-class on OpenNeuro target
subjects.

## Protocol hygiene

The consensus uses:

- a fitted source model;
- unlabeled target features;
- source-free predicted target probabilities.

The consensus does **not** use:

- held-out target labels;
- source feature rows during target adaptation;
- source labels during target adaptation.

Metadata records:

- `source_free_consensus=True`;
- `source_free_consensus_protocol`;
- `source_free_consensus_variants`;
- `source_free_consensus_weights`;
- `source_free_consensus_uses_target_labels=False`;
- `source_free_consensus_uses_source_rows_during_adaptation=False`;
- `source_free_consensus_valid_for_protocol_2_5=True`.

## Suggested OpenNeuro sweep

Use this as a separate Protocol 2.5 row, not as a replacement for the raw
Protocol 2 baseline:

```python
from neureptrace.decoding.source_free_consensus import SourceFreeConsensusVariant

result = fit_source_free_consensus_predict_proba(
    source_model=source_model,
    target_features=X_target_unlabeled,
    variants=[
        "source_raw",
        "balanced_topk",
        "robust_prior",
        SourceFreeConsensusVariant(
            "conservative_prior",
            {
                "max_iterations": 5,
                "pseudo_label_selection": "balanced_topk",
                "balanced_topk_per_class": 4,
                "target_prior_correction": "balanced",
                "target_prior_strength": 0.5,
                "target_prior_smoothing": 0.5,
            },
        ),
    ],
    consensus_mode="logit_mean",
    confidence_weight=1.0,
    balance_weight=1.0,
)
```

For balanced-accuracy endpoints, increase `balance_weight`; for datasets where
classes are known to be strongly imbalanced, decrease it and keep the raw source
posterior in the ensemble.
