# Dataset configs

NeuRepTrace can describe dataset inputs and workflow roles in a declarative YAML or JSON file. The config layer is meant to capture stable dataset structure: roots, participant IDs, file roles, loader names, metadata files, and workflow train/test roles. It does not replace loader implementations; modality-specific interpretation still belongs in Python loader code.

## Minimal YAML example

```yaml
version: 1

dataset:
  id: example_meg
  root_env: NEUREPTRACE_DATA_ROOT
  participants: "1-4,6,8"
  metadata:
    label_column: condition
    group_column: subject
  files:
    main:
      pattern: "sub-{participant}/epochs.fif"
      loader: mne_epochs
      metadata: "sub-{participant}/metadata.csv"
    calibration:
      pattern: "sub-{participant}/localizer-epo.fif"
      loader: mne_epochs

workflow:
  name: stimulus_identity
  train: main
  test: main
  label: condition
```

Use `root` for a path stored in the config file, or `root_env` for an environment variable that points to the dataset root. Relative paths are resolved against the config file location and then against the dataset root.

Participant strings support compact ranges such as `"1-4,6,8"`. Zero-padded ranges preserve padding, so `"01-03"` becomes `01`, `02`, and `03`.

## Validation

```bash
neureptrace-validate-config config.yml
neureptrace-validate-config config.yml --list-files
neureptrace-validate-config config.yml --check-files
neureptrace-validate-config config.yml --json
```

Structural validation checks that dataset file roles exist, workflow roles refer to known files, and either workflow labels or dataset metadata label columns are declared. `--check-files` additionally materializes paths and verifies that configured data, metadata, and events files exist.

## Boundary

Project packages such as PyMEGDec can register or call loaders named in the config, for example `pymegdec_fieldtrip_mat`. The config should name that loader and its file patterns, while the loader code remains responsible for reading MATLAB, FieldTrip, CTF, FIF, CSV, or other concrete formats.
