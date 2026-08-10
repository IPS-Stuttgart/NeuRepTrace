# Progressive sequence-level target fine-tuning

`neureptrace.decoding.progressive_sequence_finetune` provides a Protocol-3
neural decoder for repeated-event trials such as multi-press motor sequences.
It addresses two limitations of independent event classifiers:

1. the model can use the other presses in the same complete trial as context;
2. when the design contains one occurrence of every class per trial, prediction
   can enforce that one-to-one assignment without using evaluation labels.

The decoder trains a source backbone with held-out-source-subject validation,
then adapts to the labeled target calibration trials in three stages:

- low-rank target adapter plus classifier head;
- final sequence block plus adapter/head;
- full-backbone fine-tuning with L2-SP regularization and optional source replay.

Small calibration sets automatically stay in the more strongly regularized
stages. The sequence backbone is a residual event encoder followed by
Transformer encoder layers. A differentiable Sinkhorn term encourages a doubly
stochastic event-by-class assignment, and inference uses a Hungarian
maximum-a-posteriori assignment.

## Leakage-safe nested calibration

Use `select_nested_trial_calibration_splits` when calibration is measured in
complete trials per stratum. It reserves the maximum calibration pool first,
makes every lower-k set a nested subset, and excludes the complete maximum pool
from evaluation at every k.

```python
from neureptrace.decoding.progressive_sequence_finetune import (
    fit_progressive_sequence_target_calibrated_decoder,
    select_nested_trial_calibration_splits,
)

splits = select_nested_trial_calibration_splits(
    target_sequence_ids,
    calibration_counts=(1, 3, 5, 10, 15, 20),
    max_per_stratum=20,
    seed=13,
    context=("target", target_subject),
)

result = fit_progressive_sequence_target_calibrated_decoder(
    source_features=X_source_trials,       # trials x 4 presses x features
    source_labels=y_source_trials,         # trials x 4 presses
    source_subjects=source_subject_ids,     # one ID per source trial
    target_features=X_target_trials,
    target_labels=y_target_trials,
    target_strata=target_sequence_ids,
    split=splits[20],
    hidden_units=96,
    num_layers=2,
    num_heads=4,
    adapter_rank=8,
    source_max_epochs=120,
    adapter_steps=80,
    last_block_steps=60,
    full_finetune_steps=60,
    source_replay_weight=0.1,
    l2sp_weight=1e-4,
    random_state=13,
)

# result.probabilities: independent event probabilities
# result.constrained_probabilities: Sinkhorn-projected probabilities
# result.independent_predictions: independent argmax labels
# result.predictions: one-to-one Hungarian trial assignments
```

`target_labels` outside `result.calibration_indices` are not used during model
fitting. Score only `result.evaluation_indices`.

## Katja Button Press MEG recommendation

For the reconstructed four-variable-finger target:

- pack presses 2–5 from each correct-order trial into one
  `trials x 4 x features` tensor;
- preserve the source-fitted scaler/PCA protocol for the first comparison;
- then compare against raw source-standardized four-window channel means,
  avoiding PCA if GPU memory permits;
- provide sequence ID only as a calibration-split stratum, never as an input
  feature;
- report independent event accuracy and permutation-constrained accuracy
  separately;
- average deterministic model seeds within target before the population
  mean/SEM, matching the existing analysis.

The permutation constraint is valid only after auditing that every included
trial contains each participant-local variable finger exactly once. Do not
apply it if repetitions or omissions are present.

### Endpoint boundary

This reconstructed target is event conditioned: press times are known, presses
2–5 are scored once each, the first press is excluded, and null/background
periods are absent. It is not the same endpoint as continuous online decoding
with sliding windows through the execution period.

Consequently, the existing 65.79% independent event accuracy and 71.01%
one-to-one constrained accuracy are **not directly comparable** with Julia's
reported 59.4% sliding-window result. Any stored numerical delta against that
reference is descriptive provenance only, not a method-superiority estimate.
See `docs/katja_online_window_alignment.md` for the alignment plan.

## End-to-end Katja feature-cache runner

The repository also provides:

```bash
python -m neureptrace.katja_finger_sequence_benchmark \
  --feature-cache /path/to/katja_finger_events.npz \
  --output-dir results/katja_finger_progressive
```

The NPZ cache must contain one row per press event:

- `features`: `events x features`, or `events x ...`; all trailing dimensions
  are flattened;
- `subjects`;
- `trial_ids` (unique within participant);
- `press_positions`;
- `sequence_ids` or `seqID`;
- either participant-local `labels` or physical `finger_codes`;
- optional `correct_order` boolean rows. If absent, all rows are assumed
  already filtered.

When only physical codes are supplied, the runner sorts each participant's four
included variable codes and maps them to classes 0–3. It keeps presses 2–5 by
default, fits `StandardScaler` and rank-capped PCA on source rows only, selects
nine deterministic source participants per target, reserves the k20 pool before
constructing nested lower-k sets, and averages the five requested seeds within
target before the population mean and SEM.

Use `--source-map-json` to reproduce an existing exact target-to-nine-sources
registry. Sequence ID is used only for calibration stratification and never
enters the model features. Outputs include per-seed, per-target, and
population-summary CSV files. The historical Julia-reference columns remain in
those outputs for reproducibility, but they must be interpreted under the
endpoint boundary above.
