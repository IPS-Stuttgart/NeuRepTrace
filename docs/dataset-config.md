# Dataset configuration

NeuRepTrace can run time-resolved decoding from JSON or YAML configs. The config
keeps dataset-specific file conventions outside Python workflow code, which makes
legacy project layouts such as PyMEGDec's `Part*Data.mat` files expressible as a
recipe rather than a separate package.

## Commands

Validate a config and all referenced input files:

```bash
neureptrace validate-dataset-config configs/bush_meg/stimulus_decoding.yml --check-files
```

Print the effective config after overrides, participant expansion, and input-file
resolution:

```bash
neureptrace validate-dataset-config configs/bush_meg/stimulus_decoding.yml \
  --set participants.ids=2 \
  --print-effective-config
```

Run the configured decoder:

```bash
neureptrace decode-from-config configs/bush_meg/stimulus_decoding.yml
```

Override individual values without copying the config:

```bash
neureptrace decode-from-config configs/bush_meg/stimulus_decoding.yml \
  --set participants.ids='[2,3,4]' \
  --set decoding.classifier=lda \
  --set preprocessing.pca_components=50
```

Override values are parsed as JSON scalars when possible, so lists should be
quoted as JSON.

## Config structure

A config has five conceptual layers:

```yaml
schema_version: neureptrace.dataset.v1

paths:
  base: cwd        # cwd | config_dir | project_root

dataset:
  name: bush_meg
  type: fieldtrip_mat
  root: ${BUSH_MEG_DATA_DIR}
  participant_file: "Part{participant}Data.mat"
  variable: data

participants:
  ids: "1-4,6,8"

validation:
  channel_policy: exact

metadata:
  columns:
    - name: stimulus_class
      index: 0
  maps:
    stimulus_class:
      1: face
      2: object
  filters:
    - column: stimulus_class
      include: [1, 2]

preprocessing:
  tmin: -0.2
  tmax: 0.6
  window_size: 0.1
  window_step: 0.05

decoding:
  label_column: stimulus_class
  group_column: participant
  classifier: multiclass-svm

outputs:
  base_dir: results/bush_meg
  summary_csv: stimulus_summary.csv
  observations_csv: stimulus_observations.csv
  provenance: true
```

`dataset` describes how files are found and read. `metadata` maps source
metadata into named columns, optionally recodes values, and can filter trials.
`preprocessing` controls windowing and normalization. `decoding` selects labels,
grouping, classifiers, calibration, and tuning. `outputs` names the generated
CSVs and provenance sidecars.

## Path semantics

Input files are resolved relative to `dataset.root` when that key is present,
otherwise relative to the config file directory. Output paths are intentionally
separate: relative output paths are resolved against `outputs.base_dir` when it
is configured, or against `paths.base` otherwise.

`paths.base` supports:

- `cwd`: the current working directory; this is the default for output paths.
- `config_dir`: the directory containing the YAML/JSON config.
- `project_root`: the nearest parent containing `pyproject.toml` or `.git`, with
  the current working directory as fallback.

This prevents a config under `configs/bush_meg/` from unexpectedly writing to
`configs/bush_meg/results/...`.

## Dataset types

### `mne_epochs`

Use this when the data already exists as an MNE Epochs FIF file:

```yaml
dataset:
  type: mne_epochs
  epochs: data/sub-01_epo.fif
  metadata_csv: data/sub-01_events.csv
```

If the epochs file already contains metadata, `metadata_csv` can be omitted. When
`--check-files` is used, both the epochs file and `metadata_csv` are checked.

### `fieldtrip_mat`

Use this for FieldTrip-like MATLAB structs with `trial`, `time`, `label`, and
`trialinfo` fields:

```yaml
dataset:
  type: fieldtrip_mat
  root: ${BUSH_MEG_DATA_DIR}
  participant_file: "Part{participant}Data.mat"
  variable: data
  fields:
    trial: trial
    time: time
    label: label
    trialinfo: trialinfo

participants:
  ids: "1-4,6,8"
```

For main/cue transfer or other multi-file setups, use explicit files and attach
metadata to each file:

```yaml
dataset:
  type: fieldtrip_mat
  root: ${BUSH_MEG_DATA_DIR}
  files:
    - path: "Part2Data.mat"
      participant: 2
      split: main
    - path: "Part2CueData.mat"
      participant: 2
      split: cue
```

Extra keys on each file entry become metadata columns after loading.

## Channel alignment

`validation.channel_policy` controls multi-file concatenation:

- `exact`: all files must have identical channel names in identical order.
- `intersection`: keep only channels present in every file, in the first file's
  order, and record dropped channels in provenance.
- `first_dataset`: keep the first file's channel set and order; later files may
  have extra channels, but must contain all first-file channels.

`exact` is the recommended default for scientific reproducibility.

## Provenance

Config-driven workflows write sidecar files such as:

```text
stimulus_decoding_summary.csv.provenance.json
```

The sidecar records the config path, effective config hash, input files, optional
input-file SHA-256 hashes, and the fixed random seed used by current decoding
workflows. Disable sidecars with:

```yaml
outputs:
  provenance: false
```

Disable input hashing for very large files with:

```yaml
outputs:
  hash_input_files: false
```
