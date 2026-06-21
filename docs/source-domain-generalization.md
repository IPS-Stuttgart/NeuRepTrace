# Source-only domain generalization

NeuRepTrace supports Protocol 1 / Category 1 source-only domain generalization through `neureptrace.decoding.source_domain_generalization`.

These estimators use:

- source features `X_s`
- source labels `y_s`
- source domain IDs, usually source subject IDs

They do **not** accept held-out target features `X_t` during fitting and do **not** accept target labels `y_t`. They are therefore benchmark-valid for strict LOSO evaluations when source domains are the training subjects.

## Implemented strategies

### `subject_adversarial`

`TorchSourceAdversarialClassifier` learns a class-predictive representation while a gradient-reversal domain head tries to remove source-subject information from the embedding. This is a source-only analogue of DANN: source labels supervise the class head, and source subject IDs supervise the adversarial nuisance head.

### `group_dro`

`TorchGroupDROClassifier` minimizes robust source-domain risk. During training, source subjects with higher current loss receive larger optimization weight, reducing dependence on an easy or overrepresented source subject.

### `erm`

`TorchSourceERMClassifier` is the same neural architecture without adversarial or GroupDRO reweighting. It provides a controlled source-only baseline for the domain-generalization models.

## Domain-aware early stopping

All three strategies prefer held-out source-domain validation for early stopping. NeuRepTrace first tries to reserve one or more source subjects as validation domains. It falls back to stratified row validation only when no held-out-domain split can preserve usable class coverage.

This is stricter than ordinary random validation because it estimates whether the source-only representation generalizes across training subjects rather than only across rows from the same subjects.

## Direct use

```python
from neureptrace.decoding.source_domain_generalization import (
    fit_source_domain_generalization_predict_proba,
)

result = fit_source_domain_generalization_predict_proba(
    strategy="group_dro",
    source_features=X_source,
    source_labels=y_source,
    source_domains=source_subject_ids,
    test_features=X_heldout_subject,
)

probabilities = result.probabilities
metadata = result.metadata
```

The returned metadata records the active strategy, source-validation mode, source-domain count, and the fact that no target features or target labels were used for fitting.
