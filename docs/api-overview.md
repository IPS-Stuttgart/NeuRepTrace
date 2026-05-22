# API Overview

NeuRepTrace exposes modules for metadata preparation, manifest validation, MNE time decoding, result aggregation, calibration reporting, plotting, inference, paired decoder statistics,
probability-trace onset detection, continuous raw-stream stimulus scanning, stream-level stimulus event detection, onset chunk validation, multi-task onset workflows, onset sensitivity
analysis, sticky temporal modeling, temporal posterior smoothing, emission comparison, semantic-stage analysis, and the calibration-aware temporal-state workflow.

Key command-line modules include:

- neureptrace.metadata
- neureptrace.validate_manifest
- neureptrace.mne_time_decode_foldlocal
- neureptrace.mne_time_decode
- neureptrace.mne_time_decode_ensemble
- neureptrace.benchmark
- neureptrace.continuous_stimulus_scan
- neureptrace.results
- neureptrace.report
- neureptrace.calibration
- neureptrace.plot_time_decode
- neureptrace.plot_calibration
- neureptrace.inference
- neureptrace.paired_stats
- neureptrace.onset_detection
- neureptrace.stimulus_detection
- neureptrace.onset_validation
- neureptrace.onset_workflow
- neureptrace.onset_sensitivity
- neureptrace.temporal_model
- neureptrace.temporal_smoothing
- neureptrace.emission_compare
- neureptrace.semantic_stages
- neureptrace.temporal_state_workflow

MNE time decoding is exposed in three command flavors. The installed
`neureptrace-mne-time-decode` command and grouped `neureptrace mne-time-decode`
subcommand use `neureptrace.mne_time_decode_foldlocal`, which fits subject-level
normalization inside each outer cross-validation train fold. The historical base
implementation remains available as `neureptrace-mne-time-decode-base` and
`neureptrace mne-time-decode-base`. Calibrated logistic/linear-SVM probability
ensembling is available as `neureptrace-mne-time-decode-ensemble` and
`neureptrace mne-time-decode-ensemble`.

The base decoder also exposes `--time-decode-backend mne` for same-time
decoding. This backend uses `mne.decoding.SlidingEstimator` over NeuRepTrace's
windowed feature tensors and is intended as a parity path while replacing
hand-written per-window estimator loops with MNE-maintained decoding machinery.

Reusable table-oriented APIs include:

- `neureptrace.metrics` for calibration/probabilistic scoring metrics, pre/post window comparisons, and confusion-table summaries.
- `neureptrace.decoding.alignment_window` for applying projections fitted on one M/EEG feature window to matching-channel features from another window.
- `neureptrace.continuous_stimulus_scan` for training an event-locked decoder on one raw run, scanning a held-out raw run, exporting long-stream class probabilities, and scoring detected events.
- `neureptrace.stimulus_detection` for detecting zero, one, or many stimulus events in long probability streams and evaluating them against annotation tables.
- `neureptrace.results` for time-decoding aggregation, participant/window result tables, and peak-window diagnostics.
- `neureptrace.temporal_smoothing` for converting held-out probability traces into sticky forward-backward temporal posterior observations and corresponding fold/time metrics.
