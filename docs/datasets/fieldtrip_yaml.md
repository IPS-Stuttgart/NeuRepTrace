# Dataset YAML workflow

NeuRepTrace dataset configs describe how subject recordings map to decoding workflows. This is intended to replace hard-coded dataset conventions such as `Part10Data.mat` inside project-specific packages.

A minimal FieldTrip-MAT configuration looks like this:

```yaml
schema_version: neureptrace.dataset.v1

dataset:
  id: bush_meg
  root: "D:/Uni-Data/Bush_MEG-Data/MEG-Data"

loader:
  format: mat
  structure: fieldtrip_raw
  root_path: [data, 0]
  label_base: 1
  trim_overlong_labels: true
  ch_type: grad
  label_column: condition

participants:
  include: [10]

recordings:
  main:
    pattern: "Part{participant}Data.mat"
  cue:
    pattern: "Part{participant}CueData.mat"

analyses:
  - name: stimulus_main_cv
    task: time_decode
    recording: main
    output_csv: "outputs/{analysis}_{subject}.csv"
    observations_out: true
    window_ms: 100
    step_ms: 50
    decoder: linear_svm
    emission_mode: calibrated
    feature_preprocessor: pca_whiten
    pca_components: 0.99
    normalization: subject_baseline_whiten
    baseline_window: [-0.35, -0.05]
```

Run it with:

```bash
neureptrace dataset configs/bush_meg.yml --participant 10
```

or inspect the expanded runs without executing decoders:

```bash
neureptrace dataset configs/bush_meg.yml --dry-run
```

Relative recording paths are resolved against `dataset.root`. Relative output paths are resolved against the config file directory. Use `{participant}`, `{subject}`, `{analysis}`, `{recording}`, and `{dataset}` placeholders in path templates to avoid output collisions.

`input_format: fieldtrip-mat` requires the FieldTrip MAT loader module. Without it, use `input_format: mne-epochs` and point recordings to staged MNE Epochs FIF files plus optional `metadata_csv` files.
