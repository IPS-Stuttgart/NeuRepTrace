# BUSH-MEG source-only temporal pooling

This patch adds a source-only temporal-pooling decoder mode for the BUSH-MEG
main-task LOSO benchmark.

## Why this is worth testing

Cue-fitted alignment and calibration have repeatedly underperformed the
source-only baseline. The new mode therefore does not use `Part*CueData.mat` and
does not fit any transform on the held-out subject. Instead, it increases the
effective source training set by treating multiple nearby post-stimulus windows
from the source subjects as temporal augmentations of the same visual-class
decision problem.

Concretely, with `temporal_train_mode: pooled`, NeuRepTrace:

1. keeps the outer grouped split unchanged, so a held-out subject remains
   completely held out;
2. extracts features for all configured test-time windows;
3. stacks only the selected source-subject train windows from
   `temporal_train_window`;
4. fits one classifier on the pooled source rows; and
5. evaluates that classifier at every test-time window.

This is cheaper than the existing train-window ensemble because it fits one
classifier per fold instead of one classifier per selected train-time window.
It also targets the main failure mode seen in small cross-subject M/EEG
decoding: a high-variance source-only classifier trained on a single, somewhat
arbitrary 100-200 ms window.

## Recommended smoke run

```bash
export BUSH_MEG_DATA_DIR=/path/to/bush_meg_mat_files
neureptrace-decode-from-config configs/bush_meg/stimulus_decoding_temporal_pool.yml
```

Primary comparison metric:

```text
mean balanced_accuracy over folds at the best post-stimulus time
```
