# Semi-supervised LoRA few-shot calibration

NeuRepTrace exposes a Protocol 3 decoder in
`neureptrace.decoding.lora_few_shot` for experiments where a held-out target
subject contributes a small labeled calibration subset and, optionally, an
additional unlabeled target calibration pool.

The implementation is intentionally feature-matrix based: it accepts source rows,
source labels, target rows, and target labels for calibration-split construction.
It then trains a small source neural classifier, optionally meta-initializes its
adapter/head parameters with source-subject episodes, freezes the source backbone,
and adapts only the low-rank LoRA adapter plus the classifier head on the target
calibration subset.

## Protocol category

The standard mode uses:

- source features and labels: `X_s, y_s`;
- labeled target calibration rows: `X_t^calib, y_t^calib`;
- disjoint target evaluation features for scoring: `X_t^eval`.

It is therefore reported as
`semi_supervised_lora_few_shot_calibration` in category
`3_supervised_calibrated_target_alignment`. It is not a strict source-only result
and it is not unlabeled target adaptation.

If `target_unlabeled_features` is supplied, the target adaptation loss can also
include label-free entropy minimization and consistency regularization on that
unlabeled pool. Those rows should be separate from the scored evaluation rows for
the cleanest deployment-style Protocol 3 benchmark.

`use_evaluation_features_as_unlabeled=True` is also supported for explicitly
transductive experiments. In that mode, evaluation features are used without
labels during adaptation, and metadata records
`lora_few_shot_transductive_evaluation_features=True`. Report those results
separately from non-transductive few-shot calibration.

## Minimal usage

```python
from neureptrace.decoding.lora_few_shot import fit_lora_few_shot_target_calibrated_decoder

result = fit_lora_few_shot_target_calibrated_decoder(
    source_features=X_source,
    source_labels=y_source,
    source_subjects=source_subject_ids,  # optional, enables source-subject meta episodes
    target_features=X_target,
    target_labels=y_target,
    per_class=4,
    meta_epochs=5,
    meta_support_per_class=2,
    meta_query_per_class=2,
    lora_rank=4,
    entropy_loss_weight=0.01,
    target_unlabeled_features=X_target_unlabeled,
    seed=13,
)

# Score only the disjoint evaluation rows.
y_eval = y_target[result.evaluation_indices]
probabilities = result.probabilities
metadata = result.metadata
```

## What is adapted

Source training fits the whole small neural classifier. Target calibration then
freezes the base hidden layer and updates only:

- the low-rank LoRA matrices `lora_a` and `lora_b`;
- the classifier head, unless `adapt_classifier_head=False`.

This gives a compact few-shot target adapter without changing the source-trained
backbone.

## Semi-supervised losses

The target adaptation objective always includes supervised cross-entropy on the
labeled target calibration subset. When unlabeled target features are supplied,
it can also include:

- entropy minimization via `entropy_loss_weight`;
- prediction-consistency regularization under feature noise via
  `consistency_loss_weight` and `consistency_noise_std`;
- optional source replay via `source_replay_weight` to reduce catastrophic
  forgetting.

Target evaluation labels are never used for fitting or model selection. The test
suite checks this by perturbing evaluation labels while keeping the calibration
labels and split fixed; predictions remain identical.

## Reporting hygiene

Report these metadata columns whenever this helper is used:

- `lora_few_shot_protocol`;
- `lora_few_shot_protocol_category`;
- `lora_few_shot_uses_target_features`;
- `lora_few_shot_uses_target_labels`;
- `lora_few_shot_uses_unlabeled_target_features`;
- `lora_few_shot_uses_evaluation_features_as_unlabeled`;
- `lora_few_shot_n_target_calibration_rows`;
- `lora_few_shot_n_target_evaluation_rows`;
- `lora_few_shot_meta_episodes_run`.

Plot this method with Protocol 3 calibrated methods, not with strict Protocol 1
zero-calibration baselines.
