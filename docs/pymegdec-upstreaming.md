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
- Fold-local supervised low-rank PLS LOSO utilities.
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
