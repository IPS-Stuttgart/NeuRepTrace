# Temporal Generalization

NeuRepTrace supports temporal generalization in two complementary ways:

1. a dataset-independent matrix API for training at one time and testing at all
   other times; and
2. an MNE time-decoding ensemble mode that can directly improve reported
   time-resolved results.

For the MNE workflow, pass `--temporal-train-window START STOP` to train one
model for each decoding-window center inside the selected interval. Each model
is evaluated at every test time, and NeuRepTrace averages the resulting class
probabilities before computing accuracy, log loss, Brier score, ECE,
calibration bins, and probability-observation exports.

```bash
neureptrace-mne-time-decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column condition \
  --group-column session \
  --temporal-train-window 0.12 0.25 \
  --out results/sub-01_temporal_ensemble.csv \
  --observations-out results/sub-01_temporal_ensemble_observations.csv
```

When the discriminative latency is only approximately known, use
`--temporal-selection-window START STOP` instead. For every outer fold,
NeuRepTrace ranks candidate train-time windows by inner cross-validation on the
outer training trials only, refits the selected top-k train-time models on the
full outer train fold, and probability-ensembles them across all test times.
This keeps the held-out fold untouched while allowing a wider time grid.

```bash
neureptrace-mne-time-decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column condition \
  --group-column session \
  --temporal-selection-window 0.08 0.30 \
  --temporal-selection-top-k 3 \
  --temporal-selection-cv-splits 3 \
  --temporal-selection-metric accuracy \
  --out results/sub-01_temporal_selected.csv \
  --observations-out results/sub-01_temporal_selected_observations.csv
```

The ensemble can be combined with nested decoder tuning:

```bash
neureptrace-mne-time-decode \
  --epochs path/to/sub-01_epo.fif \
  --metadata-csv path/to/sub-01_events.csv \
  --label-column condition \
  --group-column session \
  --temporal-train-window 0.12 0.25 \
  --tune-hyperparameters \
  --tuning-cv-splits 2 \
  --tuning-scoring balanced_accuracy \
  --out results/sub-01_temporal_ensemble_tuned.csv
```

The emitted result and observation tables include provenance columns such as
`temporal_mode`, `train_time`, `test_time`, `train_window_start`,
`train_window_stop`, `temporal_train_window_start`,
`temporal_train_window_stop`, and `n_train_windows`. Nested temporal selection
also records `temporal_selection_window_start`, `temporal_selection_window_stop`,
`temporal_selection_metric`, `temporal_selection_top_k`,
`temporal_selected_train_times`, and `temporal_selection_scores`, so the selected
train-time grid remains auditable without depending on any task-specific names.

::: neureptrace.decoding.temporal_generalization
