# Covariance Features

NeuRepTrace exposes dataset-independent covariance feature helpers for M/EEG workflows through `neureptrace.decoding.covariance_features`. These helpers are usable from project-specific compatibility repositories without importing a BUSH-MEG CLI module.

The feature extractor supports the PyMEGDec-compatible representations used by the BUSH-MEG covariance LOSO workflow:

- `logeuclidean_covariance`
- `covariance_upper`
- `correlation_upper`
- `variance`

For a single trial shaped `channels x time`, use `covariance_feature_vector`. For a full array shaped `trials x channels x time`, use `window_covariance_features` with a `CovarianceWindow`.

Reusable entry points:

- `CovarianceWindow`
- `covariance_feature_vector`
- `window_covariance_features`
- `normalize_covariance_feature_mode`
- `trial_covariance`
- `channel_subset_indices`
