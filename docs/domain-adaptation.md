# Category-2 domain adaptation

NeuRepTrace's torch DANN decoder now supports optional maximum mean discrepancy (MMD) losses for unlabeled target adaptation.

The supported loss weights are:

- `domain_loss_weight`: adversarial source/target domain loss.
- `mmd_loss_weight`: marginal MMD between source and target embeddings.
- `conditional_mmd_loss_weight`: soft class-conditional MMD using source labels and target predicted probabilities.

These modes use source features, source labels, and unlabeled target features. They do not accept target labels, so they remain Protocol 2 / Category 2 methods.

Example direct use:

```python
from neureptrace.decoding.dann import fit_dann_predict_proba

result = fit_dann_predict_proba(
    source_features=X_source,
    source_labels=y_source,
    target_features=X_target,
    domain_loss_weight=0.0,
    mmd_loss_weight=0.1,
    conditional_mmd_loss_weight=0.1,
)
```

The returned metadata records the active protocol and the exact loss weights under the existing `dann_*` provenance keys.
