# YAML/JSON Dataset Configs

NeuRepTrace still accepts the existing CSV benchmark manifest.  YAML/JSON dataset configs add one layer above it: they describe dataset roots, participant IDs, file roles, loader names, and workflow defaults, then compile MNE-epochs workflows into the CSV format consumed by `neureptrace-benchmark`.

This is intended as a neutral upstream interface.  Dataset-specific packages can keep private file conventions downstream while exposing reusable manifests upstream.

## Minimal MNE example

```yaml
version: 1

dataset:
  id: nod
  root: data/nod
  participants: "01-05"
  subject_template: "sub-{participant}"
  files:
    epochs:
      pattern: "{subject}_epo.fif"
      loader: mne_epochs
    events:
      pattern: "{subject}_events.csv"
      loader: csv_events

workflow:
  name: animate
  kind: mne_time_decode
  epochs: epochs
  events: events
  label_column: condition
  group_column: run
  source_column: stim_is_animate
  positive_pattern: "True"
  positive_label: animate
  negative_label: inanimate
  options:
    n_splits: 5
    window_ms: 20
    step_ms: 10
```

Validate the config and write a CSV benchmark manifest:

```bash
neureptrace-validate-config configs/nod_animate.yml \
  --check-files \
  --write-benchmark-manifest benchmarks/nod_animate.csv
```

The grouped CLI can call the same module:

```bash
neureptrace validate-config configs/nod_animate.yml \
  --write-benchmark-manifest benchmarks/nod_animate.csv
```

Then run the existing benchmark workflow unchanged:

```bash
neureptrace-benchmark benchmarks/nod_animate.csv \
  --out-dir results/nod_animate \
  --chance 0.5
```

## Dataset roots

`dataset.root` is resolved relative to the config file unless it is absolute.  Environment variables can be used with shell-style placeholders:

```yaml
dataset:
  root: "${NRT_DATA_ROOT}"
```

A shorthand `root_env` is also accepted:

```yaml
dataset:
  root_env: NRT_DATA_ROOT
```

## Participants and placeholders

`participants` can be a list or a comma-separated string.  Inclusive integer ranges are expanded, and zero padding is preserved:

```yaml
participants: "01-03,10"
```

File patterns may use these placeholders:

- `{participant}`: the participant token after range expansion, for example `01`.
- `{participant_int}`: integer form for numeric participants, useful for format specs such as `{participant_int:02d}`.
- `{subject}`: value from `subject_template`.
- `{dataset}`: dataset ID.
- `{role}`: file role name.
- `{workflow}`: workflow name when available.

## Generic downstream adapters

For dataset-specific loaders, keep the loader name declarative and implement the actual adapter downstream:

```yaml
dataset:
  id: pymegdec
  root_env: PYMEGDEC_DATA_DIR
  participants: "1-4,6,8,9,10,13-27"
  files:
    main:
      pattern: "Part{participant}Data.mat"
      loader: pymegdec_fieldtrip_mat
    cue:
      pattern: "Part{participant}CueData.mat"
      loader: pymegdec_fieldtrip_mat

workflow:
  name: stimulus_identity_transfer
  kind: cross_subject_transfer
  train: main
  test: cue
  label_column: stimulus_id
```

Such a config can be validated for structure and file existence.  Only `kind: mne_time_decode` workflows with an `mne_epochs` epochs role are compiled into the existing benchmark CSV schema.
