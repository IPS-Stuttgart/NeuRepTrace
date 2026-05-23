# PyMEGDec upstreaming notes

PyMEGDec is now a legacy compatibility repository. Reusable decoding, dataset,
diagnostic, and reporting functionality should live in NeuRepTrace, while
paper-specific alpha-band, CTF-geometry, and historical export scripts can remain
in PyMEGDec for reproducibility.

## Already migrated

NeuRepTrace already contains the main reusable migration pieces:

- FieldTrip-style BUSH-MEG dataset loading and dataset-spec validation.
- Strict source-only BUSH-MEG LOSO decoding.
- Source-only top-k ensembles, source-fitted class-bias corrections, and
  top-k reranking from inner source-subject out-of-fold predictions.
- Generic source-OOF probability stacking for leakage-safe ensembles learned
  from source-fold observations and applied to held-out target observations.
- Fold-local supervised low-rank PLS LOSO utilities.
- Reusable PyMEGDec-compatible covariance feature extraction in
  `neureptrace.decoding.covariance_features`.
- Synthetic FieldTrip fixtures for private-data-free smoke tests.
- Generic reaction-time loading, joining, and metric-association utilities.

## Added covariance LOSO workflow

The remaining broadly reusable PyMEGDec stimulus-decoding path is now available
as:

```bash
neureptrace-bushmeg-covariance-loso configs/bush_meg/covariance_loso.yml
```

This replaces PyMEGDec's covariance-feature command with a config-driven
NeuRepTrace workflow. It loads only `Part*Data.mat` main-task files and supports
the PyMEGDec covariance representations:

- `logeuclidean_covariance`
- `covariance_upper`
- `correlation_upper`
- `variance`

The workflow performs outer held-out-subject evaluation with inner source-subject
LOSO model selection. `covariance_loso.label_shuffle_control: true` enables a
training-label shuffle null control that leaves held-out labels untouched.

For compatibility wrappers that only need feature extraction, import the generic
helpers directly instead of depending on the BUSH-MEG workflow module:

```python
from neureptrace.decoding.covariance_features import (
    CovarianceWindow,
    covariance_feature_vector,
    window_covariance_features,
)
```

## Added generic source-OOF probability stacking

The reusable part of PyMEGDec's logit-stacking workflow is the leakage boundary:
fit ensemble weights only from source-subject out-of-fold predictions, then
apply those weights to a separate held-out target table. NeuRepTrace now exposes
that idea for any probability-observation CSVs, independent of BUSH-MEG loaders
or MATLAB file conventions.

```bash
neureptrace-probability-stacking \
  --source-oof results/source_oof_observations.csv \
  --target results/heldout_target_observations.csv \
  --out results/stacked_observations.csv \
  --metrics-out results/stacked_metrics.csv
```

The grouped CLI provides the same workflow:

```bash
neureptrace probability-stacking \
  --source-oof results/source_oof_observations.csv \
  --target results/heldout_target_observations.csv \
  --out results/stacked_observations.csv
```

Use `--candidate-column` when base models are identified by a column other than
`decoder`, repeat `--candidate` to control candidate order, and repeat
`--alignment-column` when observation rows need study-specific alignment keys.
The default `stacked` weighting fits non-negative weights by minimizing
class-balanced source-OOF log loss; `uniform` and `softmax` are available as
simpler baselines. Target labels are used only for reporting prediction and
metric columns, never for fitting the source-OOF weights.