# Dataset specs

NeuRepTrace dataset specs move study-specific path, participant, and role conventions out of Python modules and into a versioned YAML or JSON file. Loader code still performs interpretation: MATLAB parsing, array indexing, metadata joins, feature extraction, MNE object creation, and shape validation remain Python responsibilities.

The intended boundary is:

- the spec describes where files are and how those files map to logical roles;
- loader adapters read those files and normalize them to NeuRepTrace inputs;
- decoding, calibration, temporal generalization, observations, and result tables stay in NeuRepTrace workflows.

## Example

```yaml
schema_version: 1

dataset:
  id: pymegdec_meg
  root: "${PYMEGDEC_DATA_DIR}"
  format: matlab_struct

participants:
  ids: ["1-4", 6, 8, 9, 10, "13-27"]
  files:
    main: "Part{participant}Data.mat"
    cue: "Part{participant}CueData.mat"

roles:
  train:
    file_role: main
  validation:
    file_role: cue

matlab:
  variable: data
  index_path: [0]

features:
  data_field: trial
  time_field: time
  label_field: trialinfo
  output: trials_channels_time
```

Use `participant_number` in file templates when numeric formatting is required:

```yaml
participants:
  ids: ["01-03"]
  files:
    epochs: "sub-{participant_number:02d}_epo.fif"
```

## Validation

Validate a spec and the files it resolves to:

```bash
neureptrace dataset-spec validate examples/pymegdec/dataset.yml
```

Validate only the schema and path templates, without requiring private data files to exist:

```bash
neureptrace dataset-spec validate examples/pymegdec/dataset.yml --no-check-exists
```

Write a CSV validation report:

```bash
neureptrace dataset-spec validate examples/pymegdec/dataset.yml --report-out dataset_validation.csv
```

Inspect the concrete participant-file table:

```bash
neureptrace dataset-spec list-files examples/pymegdec/dataset.yml
neureptrace dataset-spec list-files examples/pymegdec/dataset.yml --format json
```

The standalone entry point is equivalent:

```bash
neureptrace-dataset-spec validate examples/pymegdec/dataset.yml
```

## Current schema

The current schema version is `1`.

Required top-level sections:

- `schema_version`: currently `1`.
- `dataset`: contains `id`, `root`, and `format`.
- `participants`: contains `ids` and `files`.

Optional top-level sections:

- `roles`: maps logical roles such as `train` and `validation` to entries in `participants.files`. When omitted, every file role becomes its own logical role.
- `matlab`: MATLAB loader options. The included MATLAB adapter supports `variable`, `index_path`, `squeeze_first_element`, `squeeze_me`, and `struct_as_record`.
- `features`, `decoding`, and `outputs`: declarative workflow hints for downstream adapters and scripts.

Participant IDs may be integers, strings, compact ranges such as `"1-4"`, or mapping forms such as `{range: ["08", "10"]}` and `{id: "sub-01"}`.
