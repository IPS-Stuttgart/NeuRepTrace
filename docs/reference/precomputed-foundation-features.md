# Precomputed foundation features

`neureptrace.decoding.precomputed_foundation` supports a dependency-light frozen-feature workflow for external EEG/MEG foundation encoders.

Use this when BENDR, LaBraM, EEGPT, CBraMod, or a project-local encoder has already exported a row-aligned feature table. NeuRepTrace then aligns rows by id and trains a source-label probe without importing the upstream model package.

Supported table formats:

- `.npz` with a feature matrix and optional row ids,
- `.npy` feature matrix with sequential row ids,
- `.csv` / `.tsv` with a row-id column and numeric feature columns.

The module does not accept held-out target labels in the probe API. The `feature_fit_scope` metadata records whether the external feature extractor was declared as strict source-only, unlabeled target-adaptive, target-calibrated, or oracle/target-included.

::: neureptrace.decoding.precomputed_foundation
    options:
      members:
        - PrecomputedFoundationFeatureTable
        - PrecomputedFoundationProbeResult
        - load_precomputed_foundation_features
        - make_precomputed_foundation_feature_table
        - align_precomputed_foundation_features
        - fit_precomputed_foundation_probe
        - normalize_feature_fit_scope
