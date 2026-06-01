# Dataset specs

NeuRepTrace dataset specs are versioned YAML or JSON files that describe how to find dataset files and metadata without hard-coding project conventions in Python modules.

Use specs for dataset-level concerns:

- data-root resolution;
- subject or participant selection;
- split naming, such as `main`, `cue`, `train`, or `test`;
- file path templates;
- metadata CSV templates;
- label and group columns;
- loader options for MNE, MATLAB FieldTrip-like data, or CSV feature matrices;
- workflow defaults that should be merged into generated manifests.

Keep scientific behavior in Python workflows. Filtering, window extraction, dimensionality reduction, classifier fitting, calibration, alpha metrics, CTF geometry transforms, and statistical summaries should remain tested Python code rather than YAML logic.

## CLI

Validate a spec and inspect the resolved files:

```bash
neureptrace dataset validate examples/configs/pymegdec_bushmeg.yml
```

Write the inventory to CSV and fail when resolved files are missing:

```bash
neureptrace dataset validate examples/configs/pymegdec_bushmeg.yml \
  --require-files \
  --out results/bushmeg_inventory.csv
```

Expand an MNE-compatible spec into a benchmark-style manifest:

```bash
neureptrace dataset manifest configs/nod.yml \
  --workflow animate \
  --out NeuRepTrace-Paper/benchmarks/nod_animate_from_spec.csv
```

For MATLAB/FieldTrip-like private datasets, manifest expansion is still useful as an inventory and provenance table. Direct model workflows should call `load_split_dataset()` or a workflow-specific adapter.

## Minimal schema

```yaml
schema_version: neureptrace.dataset.v1
dataset_id: example_meg

root:
  path: data/example

subjects:
  include: "1-3,5"

splits:
  epochs:
    loader: mne_epochs
    path_template: "sub-{subject02d}_epo.fif"
    metadata_template: "sub-{subject02d}_events.csv"
    label_column: condition
    group_column: run

workflows:
  benchmark:
    split: epochs
    manifest:
      n_splits: 5
      decoder: logistic
```

## Root resolution

`root` supports three mechanisms, checked in this order:

1. `path`: a root path, relative to the spec file when not absolute;
2. `env`: an environment variable containing the root path;
3. `fallback_file`: a text file containing the root path, also relative to the spec file when not absolute.

When none are set, paths are resolved relative to the spec file directory.

## Subject placeholders

`subjects.include` accepts comma-separated numbers and ranges, for example `1-4,6,8`. Path templates can use:

- `{subject}` or `{participant}` for the literal subject token;
- `{subject_int}` or `{participant_int}` for integer subjects;
- `{subject02d}` or `{participant02d}` for zero-padded integer subjects.

## Loaders

### `mne_epochs`

Use for existing NeuRepTrace MNE workflows. The resolved data path is exported as the `epochs` column in generated manifests.

### `matlab_fieldtrip`

Use for FieldTrip-style MATLAB structs with fields such as `trial`, `time`, `label`, and `trialinfo`. The loader returns a canonical `TrialDataset` with shape `trials x channels x time`.

Relevant split keys:

```yaml
loader: matlab_fieldtrip
path_template: "Part{subject}Data.mat"
mat_key: data
trial_key: trial
time_key: time
channel_key: label
label_key: trialinfo
label_index_base: 1
trial_layout: channels_by_time
```

For private datasets with paired participant files, `dataset_config` also
supports a participant-expanded mapping of logical file roles to templates:

```yaml
dataset:
  type: fieldtrip_mat
  root: ${BUSHMEG_DATA_DIR}
  file_templates:
    main: "Part{participant}Data.mat"
    cue: "Part{participant}CueData.mat"
participants:
  ids: "1-4,6,8,9,10,13-27"
```

Each loaded file receives `participant`, `split`, and `file_role` metadata so
train/test transfer configs can select `split: main` and `split: cue` without a
dataset-specific Python wrapper.

### `csv_feature_matrix`

Use for a wide numeric CSV where rows are trials and numeric columns are features. A configured `label_column` is removed from the feature matrix and returned as labels.

## PyMEGDec-style example

The example config in `examples/configs/pymegdec_bushmeg.yml` captures the `Part{subject}Data.mat` and `Part{subject}CueData.mat` conventions without hard-coding them in a package module. It is intended as the migration target for stimulus-transfer loading. Alpha, reaction-time, and CTF-geometry-specific analyses can remain in a compatibility wrapper until they are generalized.
