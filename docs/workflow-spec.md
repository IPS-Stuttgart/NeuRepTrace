# Workflow Specs

NeuRepTrace workflow specs describe the dataset roles, metadata tables, feature/model settings, evaluation split, and output table names needed by a workflow without embedding project-specific file conventions in NeuRepTrace.

The schema is intentionally declarative.  A dataset package such as PyMEGDec should still provide the concrete loader implementation; the upstream spec records the loader name and the files or patterns it should consume.

## Commands

Validate a JSON config:

```bash
neureptrace-validate-workflow workflow.json
```

Validate and check declared paths:

```bash
neureptrace workflow validate workflow.yml --check-files
```

Print the normalized representation or the JSON Schema:

```bash
neureptrace workflow show workflow.json
neureptrace workflow schema > neureptrace-workflow.schema.json
```

YAML files require PyYAML to be installed.  JSON works with the Python standard library.

## Example

```yaml
version: 1

workflow:
  kind: cross_subject_decoding
  name: pymegdec_main_to_cue_stimulus

dataset:
  id: pymegdec_main_cue
  root: ${PYMEGDEC_DATA_DIR}
  participants: [1, 2, 3, 4]
  files:
    main:
      loader: pymegdec.fieldtrip_mat
      pattern: Part{participant}Data.mat
      metadata: stimulus
    cue:
      loader: pymegdec.fieldtrip_mat
      pattern: Part{participant}CueData.mat
      metadata: stimulus
  metadata:
    stimulus:
      path: stimulus_metadata.csv
      key: stimulus_id

features:
  label: stimulus_id
  window:
    size_s: 0.1
    centers_s:
      start: -0.2
      stop: 0.6
      step: 0.05

normalization:
  name: subject_baseline_z
  baseline_window_s: [-0.2, 0.0]

model:
  classifier: multiclass-svm
  params:
    pca_components: 100

evaluation:
  split: leave_one_subject_out
  train: main
  test: cue

outputs:
  root: outputs/pymegdec_main_cue
  tables:
    predictions: predictions.csv
    summary: summary.csv
```

## Validation scope

The workflow-spec validator checks the portable schema: top-level version, workflow kind, dataset identity, data-file roles, loader names, and optional metadata declarations.  With `--check-files`, it also checks concrete paths and expands list-valued participant patterns such as `Part{participant}Data.mat`.

The validator does not import loader implementations, read M/EEG files, or execute a decoder.  Those behaviors belong to workflow runners and dataset packages layered on top of the schema.
