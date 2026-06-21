# Semi-supervised LoRA few-shot calibration

NeuRepTrace exposes Protocol 3 LoRA decoders for cross-subject experiments
where a held-out target subject contributes a small labeled calibration subset
and, optionally, an unlabeled target feature pool. Report these results as
supervised calibrated target-alignment/adaptation results, not as strict
source-only or unlabeled-only adaptation.

Two entry points are available:

- `neureptrace.decoding.lora_few_shot.fit_lora_few_shot_target_calibrated_decoder`
  is a feature-matrix helper that returns `lora_few_shot_*` metadata.
- `neureptrace.decoding.semi_supervised_lora_few_shot.fit_semi_supervised_lora_few_shot_decoder`
  is the semi-supervised helper that returns `semi_supervised_lora_*` and
  `few_shot_*` metadata.

## Protocol category

The standard mode uses:

- source features and labels: `X_s, y_s`;
- labeled target calibration rows: `X_t^calib, y_t^calib`;
- disjoint target evaluation features for scoring: `X_t^eval`;
- no target evaluation labels during fitting, adaptation, hyperparameter choice,
  or probability alignment.

Both helpers record the protocol as `semi_supervised_lora_few_shot_calibration`
in category `3_supervised_calibrated_target_alignment`.

If unlabeled target features are supplied, the target adaptation loss can also
include label-free entropy minimization and consistency regularization on that
unlabeled pool. Those rows should be separate from the scored evaluation rows for
the cleanest deployment-style Protocol 3 benchmark.

Using evaluation features as unlabeled inputs is supported for explicitly
transductive experiments. Report those results separately from non-transductive
few-shot calibration.

## What is fitted

The implementation has three stages:

1. A small MLP is pretrained on labeled source rows.
2. If source subject/group IDs are provided, a Reptile-style episodic pass treats
   source subjects as pseudo-target tasks and meta-learns the LoRA adapter/head
   initialization.
3. The target model freezes the base network and adapts low-rank LoRA adapters
   plus the configured classifier-head subset on the labeled target calibration
   rows. Optional unlabeled target rows can add entropy minimization and
   consistency losses.

This gives a dependency-light LoRA/meta-learning baseline without requiring an
external foundation model.

## Feature-matrix helper

```python
from neureptrace.decoding.lora_few_shot import (
    fit_lora_few_shot_target_calibrated_decoder,
)

result = fit_lora_few_shot_target_calibrated_decoder(
    source_features=X_source,
    source_labels=y_source,
    source_subjects=source_subject_ids,  # optional, enables source-subject episodes
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

## Semi-supervised helper

```python
from neureptrace.decoding.semi_supervised_lora_few_shot import (
    fit_semi_supervised_lora_few_shot_decoder,
)

result = fit_semi_supervised_lora_few_shot_decoder(
    source_features=X_source,
    source_labels=y_source,
    source_groups=source_subject_ids,  # optional but enables source-subject episodes
    target_features=X_target,
    target_labels=y_target,
    per_class=4,
    seed=13,
    hidden_units=64,
    lora_rank=4,
    source_pretrain_epochs=80,
    meta_epochs=20,
    target_adaptation_steps=80,
    entropy_loss_weight=0.02,
)

# Score only disjoint evaluation rows.
y_eval = y_target[result.evaluation_indices]
y_prob = result.probabilities
metadata = result.metadata
```

Set `use_evaluation_features_unlabeled=False` to adapt only from the labeled
calibration subset plus any separately supplied `extra_unlabeled_target_features`.

Report at least:

- `few_shot_protocol`;
- `few_shot_protocol_category`;
- `few_shot_n_target_calibration_rows`;
- `few_shot_n_target_evaluation_rows`;
- `semi_supervised_lora_rank`;
- `semi_supervised_lora_meta_learning_enabled`;
- `semi_supervised_lora_meta_episodes`;
- `semi_supervised_lora_transductive_evaluation_features`;
- `semi_supervised_lora_uses_unlabeled_target_features`.

## Reporting hygiene

Do not merge these results into Protocol 1 zero-calibration tables. The target
subject contributes labeled calibration rows, so this is a supervised calibrated
alignment/adaptation protocol even when unlabeled target losses are also enabled.
