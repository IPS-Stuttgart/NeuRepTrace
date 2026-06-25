# Foundation model features

NeuRepTrace can use external EEG/MEG foundation encoders as frozen feature extractors through
`neureptrace.decoding.foundation`. The integration remains dependency-light: NeuRepTrace does not
vendor BENDR, LaBraM, EEGPT, CBraMod, or other model repositories. Instead, it provides wrappers that
standardize how externally supplied TorchScript modules, trusted PyTorch modules, or state-dict
checkpoints are reshaped, frozen, pooled, and probed inside fold-local decoding workflows.

## Supported wrapper families

The built-in family names are:

| Family | Decoder alias after registration | Conservative defaults | Typical use |
| --- | --- | --- | --- |
| `generic` | `foundation-linear-probe` | `pooling="flatten"`, `input_layout="auto"` | Any project-local encoder or exported TorchScript module. |
| `bendr` | `bendr-linear-probe` | `pooling="mean_time"`, `input_layout="channels_first"` | BENDR-style backbones consuming `batch x channels x time`. |
| `labram` | `labram-linear-probe` | `pooling="cls"`, `input_layout="channels_first"` | LaBraM-style transformer exports with class-token embeddings. |
| `eegpt` | `eegpt-linear-probe` | `pooling="cls"`, `input_layout="channels_first"` | EEGPT-style transformer exports with class-token or hidden-state outputs. |
| `cbramod` | `cbramod-linear-probe` | `pooling="mean_time"`, `input_layout="channels_first"` | CBraMod-style frozen time-series backbones. |

These wrappers do not assume a single upstream API. Each model project can export slightly different
objects, so NeuRepTrace exposes explicit selectors for `output_key`, `output_attribute`,
`output_index`, `encoder_attr`, `checkpoint_key`, and `state_dict_key`.

## Direct API: TorchScript or already-constructed module

```python
from neureptrace.decoding.foundation import make_foundation_linear_probe

probe = make_foundation_linear_probe(
    {
        "model_family": "labram",       # generic, bendr, labram, eegpt, cbramod
        "model_path": "encoder.ts",     # TorchScript encoder exported from the model project
        "input_shape": [64, 100],        # reshape each flattened row to channels x time
        "pooling": "cls",               # flatten, mean_time, mean_tokens, cls, last
        "probe": "logistic",            # logistic, linear_svm, ridge, lda
        "C": 1.0,
    }
)
probe.fit(X_source, y_source)
probabilities = probe.predict_proba(X_target)
```

You can also pass an instantiated `torch.nn.Module` directly:

```python
probe = make_foundation_linear_probe(
    {
        "model_family": "bendr",
        "encoder": my_bendr_encoder,
        "input_shape": "64x256",
        "pooling": "mean_time",
    }
)
```

The encoder is placed in evaluation mode, moved to the requested device, and frozen before feature
extraction. Torch is imported lazily, so standard scikit-learn NeuRepTrace workflows are unaffected
unless this path is selected.

## Loading state-dict checkpoints

For repositories that expose Python constructors rather than TorchScript exports, provide an importable
factory and checkpoint metadata:

```python
probe = make_foundation_linear_probe(
    {
        "model_family": "eegpt",
        "load_mode": "state_dict",
        "model_factory": "my_eegpt_package.models:build_encoder",
        "model_kwargs": {"n_channels": 64, "sfreq": 250},
        "model_path": "eegpt_checkpoint.pt",
        "state_dict_key": "encoder_state_dict",  # optional; common keys are auto-detected
        "strip_state_dict_prefix": True,          # strips module./model./encoder. when all keys share it
        "input_shape": [64, 256],
        "pooling": "cls",
    }
)
```

Use `load_mode="module"` only for trusted Python pickles/checkpoints. The module loader intentionally
requests non-weights-only loading where supported by PyTorch; TorchScript and state-dict loading are
preferred for reproducible decoding runs.

## Input and output adaptation

Common input controls:

| Parameter | Meaning |
| --- | --- |
| `input_shape` | Reshape each flattened feature row before encoder inference. |
| `input_layout="channels_first"` | Keep `batch x channels x time`. |
| `input_layout="time_first"` | Convert `batch x channels x time` to `batch x time x channels`. |
| `input_layout="add_channel_dim"` | Convert to `batch x 1 x channels x time`. |
| `input_layout="add_feature_dim"` | Convert to `batch x channels x 1 x time`. |
| `preprocessing="trial_zscore"` | Stateless per-trial z-scoring over all non-batch axes. |
| `preprocessing="channel_zscore"` | Stateless per-channel z-scoring over the last axis. |

Common output controls:

| Parameter | Meaning |
| --- | --- |
| `output_key` | Select a key from dictionary outputs, including dotted or slash paths. |
| `output_attribute` | Select an attribute from HuggingFace-like output objects. |
| `output_index` | Select an index from tuple/list outputs after key/attribute selection. |
| `encoder_attr` | Select a submodule from a loaded checkpoint, for example `"encoder"` or `"backbone.encoder"`. |
| `pooling` | Convert encoder outputs to 2-D features: `flatten`, `mean_time`, `mean_tokens`, `mean_channels`, `cls`, or `last`. |

## Optional decoder registration

Foundation probes are opt-in. Register them before using the normal decoder factory:

```python
from neureptrace.decoding import make_decoder
from neureptrace.decoding.foundation import register_foundation_linear_probe

register_foundation_linear_probe()

decoder = make_decoder(
    "labram-linear-probe",
    emission_mode="uncalibrated",
    classifier_param={
        "model_path": "labram_encoder.ts",
        "input_shape": "64x100",
        "pooling": "cls",
        "probe": "linear_svm",
    },
)
```

The family-specific decoder aliases are convenience wrappers around the same frozen-encoder pipeline.
You can always use `foundation-linear-probe` with `classifier_param={"model_family": "..."}` instead.

## Precomputed feature tables

If the upstream foundation model is easier to run outside NeuRepTrace, export a row-aligned feature table
and use `neureptrace.decoding.precomputed_foundation` instead of loading Torch modules at decode time.
This path supports `.npy`, `.npz`, `.csv`, and `.tsv` tables:

```python
from neureptrace.decoding.precomputed_foundation import (
    fit_precomputed_foundation_probe,
    load_precomputed_foundation_features,
)

table = load_precomputed_foundation_features(
    "labram_features.npz",
    source_model="LaBraM",
    feature_fit_scope="external_frozen",
)

result = fit_precomputed_foundation_probe(
    feature_table=table,
    train_row_ids=source_trial_ids,
    train_labels=source_labels,
    test_row_ids=target_trial_ids,
)
```

The probe API has no `target_labels` argument. The `feature_fit_scope` field records whether the external
feature extractor was declared as `external_frozen` / `source_only`, `source_plus_unlabeled_target`,
`target_calibrated`, or `oracle_target_included`, so downstream reports can distinguish Protocol 1,
Protocol 2, Protocol 3, and oracle feature generation.

## Protocol interpretation

A frozen external encoder plus a source-label probe is **Protocol 1** when the held-out target subject
is not used for fitting, model selection, or normalization beyond ordinary fold-local evaluation. If
the encoder or an adapter is updated with unlabeled target rows, report it as **Protocol 2**. If labeled
target calibration trials are used for probing, fine-tuning, LoRA/adapters, or model selection, report
it as **Protocol 3**. Using all target labels or choosing checkpoints by target accuracy is an oracle
upper bound rather than a benchmark-valid foundation-model result.

The wrappers in this module implement the frozen-feature-extractor part. Fine-tuning, LoRA, masked
pretraining, and target-adaptive foundation-model updates should be implemented as separate Protocol 2
or Protocol 3 paths so the target-information boundary remains auditable.