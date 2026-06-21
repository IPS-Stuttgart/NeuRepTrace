# Few-shot target calibration

NeuRepTrace exposes a supervised target-calibration primitive in
`neureptrace.decoding.few_shot` for Protocol 3 experiments.  The protocol is
intended for LOSO or cross-subject benchmarks where a small, labeled calibration
subset from the held-out target subject is allowed, and all reported scores must
be computed on disjoint held-out target rows.

## Protocol category

The helper uses:

- source features and labels: `X_s, y_s`;
- a small labeled target calibration subset: `X_t^calib, y_t^calib`;
- disjoint target evaluation features for scoring: `X_t^eval`.

It therefore records the protocol as
`supervised_target_few_shot_calibration` and category
`3_supervised_calibrated_target_alignment`.  It is not strict source-only and it
is not unlabeled target adaptation.

## Minimal usage

```python
from neureptrace.decoding.few_shot import fit_few_shot_target_calibrated_decoder

result = fit_few_shot_target_calibrated_decoder(
    source_features=X_source,
    source_labels=y_source,
    target_features=X_target,
    target_labels=y_target,
    per_class=4,
    seed=13,
    decoder_name="logistic",
    emission_mode="uncalibrated",
)

# Score only the disjoint evaluation rows.
y_eval = y_target[result.evaluation_indices]
y_hat = result.probabilities.argmax(axis=1)
metadata = result.metadata
```

`select_few_shot_target_calibration_split` can be called directly when the split
must be reused across several time windows, decoders, or feature spaces.  The
splitter is deterministic for a fixed seed and context, selects the same number
of calibration rows per target class, and rejects folds where a class would have
no evaluation rows left.

## Semi-supervised LoRA/meta-learning variant

For a stronger Protocol 3 neural baseline, use
`neureptrace.decoding.semi_supervised_lora_few_shot`.  That module pretrains a
source model, optionally meta-learns a LoRA adapter initialization from
source-subject episodes, and then adapts only LoRA adapters plus the configured
classifier-head subset on the labeled target calibration rows.  It can also use
unlabeled target features through entropy/consistency losses, while still
rejecting target evaluation labels during fitting.

See `docs/semi_supervised_lora_few_shot.md` for protocol metadata and reporting
requirements.

## Reporting hygiene

Report the following metadata columns whenever this helper is used:

- `few_shot_protocol`;
- `few_shot_protocol_category`;
- `few_shot_uses_target_features`;
- `few_shot_uses_target_labels`;
- `few_shot_target_calibration_per_class`;
- `few_shot_n_target_calibration_rows`;
- `few_shot_n_target_evaluation_rows`.

Do not compare this protocol as a strict zero-calibration result.  It should be
plotted or tabulated separately from Protocol 1 and Protocol 2 results.
