# Transfer Component Analysis

`neureptrace.decoding.transfer_component_analysis` implements a dependency-light **Category 2 / unlabeled target-adaptive** Transfer Component Analysis (TCA) utility.

TCA learns a latent space from source features and unlabeled held-out target features, then a downstream classifier can be trained using source labels only. The public APIs intentionally have no `target_labels` argument.

Supported kernels:

- `linear`
- `rbf` with explicit gamma or `gamma="median"`

Typical use:

```python
from neureptrace.decoding.transfer_component_analysis import fit_tca_transfer_classifier

result = fit_tca_transfer_classifier(
    source_features=X_source,
    source_labels=y_source,
    target_features=X_target_unlabeled,
    n_components=16,
    kernel="linear",
)
```

Protocol interpretation:

\[
X_s, y_s, X_t \text{ are used}; \quad y_t \text{ is not used.}
\]

::: neureptrace.decoding.transfer_component_analysis
    options:
      members:
        - TransferComponentAnalysisModel
        - TransferComponentAnalysisResult
        - TCATransferClassificationResult
        - transfer_component_analysis_features
        - transform_with_tca_model
        - fit_tca_transfer_classifier
        - normalize_tca_kernel
