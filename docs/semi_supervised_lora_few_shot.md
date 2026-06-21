# Semi-supervised LoRA few-shot calibration

`neureptrace.decoding.semi_supervised_lora_few_shot` implements a Protocol 3
few-shot target-calibration decoder for cross-subject experiments.  It is meant
for settings where a small labeled calibration subset from the held-out target
subject is allowed and all reported metrics are computed on disjoint target
evaluation rows.

## Protocol category

The helper uses:

- source features and labels: `X_s, y_s`;
- labeled target calibration features: `X_t^calib, y_t^calib`;
- optional unlabeled target features, including the evaluation feature batch when
  `use_evaluation_features_unlabeled=True`;
- no target evaluation labels during fitting, adaptation, hyperparameter choice,
  or probability alignment.

It records the protocol as `semi_supervised_lora_few_shot_calibration` and the
category as `3_supervised_calibrated_target_alignment`.  When evaluation features
are used unlabeled, report it as a transductive/semi-supervised Category 3 result,
not as strict source-only or unlabeled-only adaptation.

## What is fitted

The implementation has three stages:

1. A small MLP is pretrained on labeled source rows.
2. If source subject/group IDs are provided, a Reptile-style episodic pass treats
   source subjects as pseudo-target tasks and meta-learns the LoRA adapter/head
   initialization.
3. The target model freezes the base network and adapts only low-rank LoRA
   adapters plus the configured classifier-head subset on the labeled target
   calibration rows.  Optional unlabeled target rows can add entropy minimization
   and consistency losses.

This gives a dependency-light LoRA/meta-learning baseline without requiring an
external foundation model.

## Minimal usage

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

## Reporting hygiene

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

Do not merge these results into Protocol 1 zero-calibration tables.  The target
subject contributes labeled calibration rows, so this is a supervised calibrated
alignment/adaptation protocol even when unlabeled target losses are also enabled.
