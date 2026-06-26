# Random subspace ensemble

`neureptrace.decoding.random_subspace` implements a strict source-only random-subspace ensemble for M/EEG feature matrices.

Each ensemble member is trained on a deterministic random subset of feature columns. Optional row bootstrapping is applied only to training rows. Test rows are only scored and never influence feature-subspace selection, row sampling, model fitting, or model weighting.

This is a **Category 1 / strict source-only** method.

Typical usage:

```python
from neureptrace.decoding.random_subspace import fit_random_subspace_ensemble

result = fit_random_subspace_ensemble(
    train_features=X_source,
    train_labels=y_source,
    test_features=X_target,
    config={"n_estimators": 32, "feature_fraction": 0.5},
)

probabilities = result.probabilities
predictions = result.predictions
```

::: neureptrace.decoding.random_subspace
    options:
      members:
        - RandomSubspaceEnsembleConfig
        - RandomSubspaceMember
        - RandomSubspaceEnsembleResult
        - fit_random_subspace_ensemble
        - random_subspace_ensemble_config
        - sample_feature_subspaces
