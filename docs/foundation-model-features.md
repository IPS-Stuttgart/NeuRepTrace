# Foundation model features

NeuRepTrace can use external EEG/MEG foundation encoders as frozen feature extractors through
`neureptrace.decoding.foundation`. The integration is intentionally lightweight: NeuRepTrace does
not vendor BENDR, LaBraM, EEGPT, CBraMod, or other model packages. Instead, pass a TorchScript encoder
or a Python `torch.nn.Module`, transform each trial/window into a latent feature vector, and train a
linear probe on the labeled source rows.

## Direct API

```python
from neureptrace.decoding.foundation import make_foundation_linear_probe

probe = make_foundation_linear_probe(
    {
        "model_path": "encoder.ts",       # TorchScript encoder from a foundation-model project
        "input_shape": [64, 100],         # reshape each flattened row to channels x time
        "pooling": "flatten",            # or mean_time, cls, last
        "probe": "logistic",             # logistic, linear_svm, ridge, lda
        "C": 1.0,
    }
)
probe.fit(X_source, y_source)
probabilities = probe.predict_proba(X_target)
```

The encoder is placed in evaluation mode, moved to the requested device, and frozen before feature
extraction. Torch is imported lazily, so standard scikit-learn NeuRepTrace workflows are unaffected
unless this path is selected.

## Optional decoder registration

The foundation probe is opt-in. Register it before using the normal decoder factory:

```python
from neureptrace.decoding import make_decoder
from neureptrace.decoding.foundation import register_foundation_linear_probe

register_foundation_linear_probe()
decoder = make_decoder(
    "foundation-linear-probe",
    emission_mode="uncalibrated",
    classifier_param={
        "model_path": "encoder.ts",
        "input_shape": "64x100",
        "pooling": "mean_time",
        "probe": "linear_svm",
    },
)
```

## Protocol interpretation

A frozen external encoder plus a source-label probe is **Protocol 1** when the held-out target subject
is not used for fitting, model selection, or normalization beyond ordinary fold-local evaluation. If
the encoder or an adapter is updated with unlabeled target rows, report it as **Protocol 2**. If labeled
target calibration trials are used for probing, fine-tuning, LoRA/adapters, or model selection, report
it as **Protocol 3**. Using all target labels or choosing checkpoints by target accuracy is an oracle
upper bound rather than a benchmark-valid foundation-model result.
