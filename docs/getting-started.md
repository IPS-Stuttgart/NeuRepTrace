# Getting Started

Install NeuRepTrace from a source checkout:

```bash
poetry install --with dev
```

Check that the environment and core dependencies are usable before launching a
benchmark:

```bash
neureptrace doctor --skip-optional
```

Dataset configs can be checked with the same diagnostic command:

```bash
neureptrace doctor --dataset-config path/to/dataset.yml --skip-optional
```

Inspect the grouped CLI before launching a workflow:

```bash
neureptrace --list-commands
```

Automation and documentation tooling can use the machine-readable inventory,
which includes each grouped command, its backing module, and equivalent aliases:

```bash
neureptrace --list-commands --list-format json
```

Run the first benchmark against an MNE epochs file with the installed
fold-local time decoder:

```bash
neureptrace-mne-time-decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column stim_is_animate \
  --group-column session \
  --out results/nod_sub-01_animate.csv
```

The default `neureptrace-mne-time-decode` command fits any subject-level epoch
normalization statistics inside each outer cross-validation train fold before
scoring held-out trials. This is the recommended path for benchmark results.
For exact comparisons to historical runs that used whole-subject normalization,
use `neureptrace-mne-time-decode-base` or the grouped command
`neureptrace mne-time-decode-base`.

For same-time decoding, the base command now uses
`mne.decoding.SlidingEstimator` by default:

```bash
neureptrace-mne-time-decode-base \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column stim_is_animate \
  --group-column session \
  --time-decode-backend mne \
  --out results/nod_sub-01_animate_mne_backend.csv
```

Use `--time-decode-backend sklearn` only when you need the previous
hand-written per-window estimator loop for historical comparisons.

Use `--metadata-csv` when the labels are stored outside the epochs metadata and
`--group-column` when cross-validation should keep sessions or runs separated.

To run the calibrated logistic/linear-SVM probability ensemble, use the dedicated
ensemble entry point:

```bash
neureptrace-mne-time-decode-ensemble \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column stim_is_animate \
  --group-column session \
  --decoder logistic-svm-ensemble \
  --out results/nod_sub-01_animate_ensemble.csv \
  --observations-out results/nod_sub-01_animate_ensemble_observations.csv
```

The same workflows are also available through the grouped CLI as
`neureptrace mne-time-decode`, `neureptrace mne-time-decode-base`, and
`neureptrace mne-time-decode-ensemble`.

If the metadata does not yet contain the binary decoding target, derive one from
a text column:

```bash
python -m neureptrace.metadata \
  --events-csv data/nod/sub-01_events.csv \
  --source-column stim_is_animate \
  --positive-pattern "True" \
  --label-column condition \
  --positive-label animate \
  --negative-label inanimate \
  --out data/nod/sub-01_metadata_animate.csv
```