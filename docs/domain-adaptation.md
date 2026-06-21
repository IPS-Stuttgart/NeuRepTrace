# Category-2 domain adaptation

NeuRepTrace supports torch-based Category-2 neural adaptation methods that use source features, source labels, and unlabeled target features. They do **not** accept target labels, so they remain Protocol 2 / Category 2 methods.

## DANN + MMD

The original `TorchDANNClassifier` supports:

- `domain_loss_weight`: marginal adversarial source/target domain loss.
- `mmd_loss_weight`: marginal maximum mean discrepancy between source and target embeddings.
- `conditional_mmd_loss_weight`: soft class-conditional MMD using source labels and target predicted probabilities.

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

The returned metadata records the active protocol and exact loss weights under the existing `dann_*` provenance keys.

## CDAN: conditional adversarial adaptation

`TorchCDANClassifier` implements Conditional Domain Adversarial Networks (CDAN) as a first-class decoder. Instead of giving the domain discriminator only the learned embedding, CDAN gives it the multilinear interaction between the embedding and the class-posterior vector. This makes the adversarial signal class-aware while still using no target labels.

Supported CDAN options:

- `cdan_loss_weight`: conditional adversarial loss on the embedding/posterior multilinear map.
- `cdan_entropy_conditioning`: entropy-based weighting of low-uncertainty examples.
- `cdan_randomized_dim`: optional randomized multilinear map dimension for high-dimensional embeddings or many classes.
- `domain_loss_weight`: optional marginal DANN loss in addition to CDAN.
- `mmd_loss_weight`: optional marginal MMD.
- `conditional_mmd_loss_weight`: optional soft class-conditional MMD.

Example:

```python
from neureptrace.decoding.cdan import fit_cdan_predict_proba

result = fit_cdan_predict_proba(
    source_features=X_source,
    source_labels=y_source,
    target_features=X_target,
    cdan_loss_weight=0.2,
    cdan_randomized_dim=256,
    conditional_mmd_loss_weight=0.05,
)
```

The returned metadata uses `cdan_*` provenance keys and explicitly records that target features were used while target labels were not.
