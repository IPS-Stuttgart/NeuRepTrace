# Dataset configuration

NeuRepTrace can run time-resolved decoding from JSON or YAML configs. The config
keeps dataset-specific file conventions outside Python workflow code, which makes
legacy project layouts such as PyMEGDec's `Part*Data.mat` files expressible as a
recipe rather than a separate package.

## Commands

Validate a config:

```bash
neureptrace validate-dataset-config configs/bush_meg/stimulus_decoding.yml --check-files
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

A config has four conceptual layers:

```yaml
schema_version: neureptrace.dataset.v1

dataset:
  name: bush_meg
  type: fieldtrip_mat
  root: ${BUSH_MEG_DATA_DIR}
  participant_file: "Part{participant}Data.mat"
  variable: data

participants:
  ids: "1-4,6,8"

metadata:
  columns:
    - name: stimulus_class
      index: 0

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
  summary_csv: results/bush_meg/stimulus_summary.csv
  observations_csv: results/bush_meg/stimulus_observations.csv
```

`dataset` describes how files are found and read. `metadata` maps source
metadata into named columns. `preprocessing` controls windowing and normalization.
`decoding` selects labels, grouping, classifiers, calibration, and tuning.
`outputs` names the generated CSVs.

## Dataset types

### `mne_epochs`

Use this when the data already exists as an MNE Epochs FIF file:

```yaml
dataset:
  type: mne_epochs
  epochs: data/sub-01_epo.fif
  metadata_csv: data/sub-01_events.csv
```

If the epochs file already contains metadata, `metadata_csv` can be omitted.

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
    sensor_geometry: grad
```

The loader expects trials in `channels × time` orientation. Set
`dataset.transpose_trials: true` if a source stores trials as `time × channels`.

## Migration from PyMEGDec

PyMEGDec's participant-file convention becomes a dataset recipe:

```yaml
dataset:
  type: fieldtrip_mat
  root: ${BUSH_MEG_DATA_DIR}
  participant_file: "Part{participant}Data.mat"
participants:
  ids: "1-4,6,8,9,10,13-27"
```

The old cue-file convention can be represented with explicit files while keeping
participant and split labels in metadata:

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

Keep reusable decoding, calibration, temporal generalization, and probability
observation export code in NeuRepTrace. Keep paper-specific alpha/RT scripts as
examples, reproducibility scripts, or a separate analysis repository.
