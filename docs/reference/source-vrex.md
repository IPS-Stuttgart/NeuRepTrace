# Source-only VREx

`neureptrace.decoding.source_vrex` implements variance risk extrapolation across source subjects or domains.

The training objective combines mean source-domain classification risk with a penalty on variation in risk across source domains. Fitting uses source features, source labels, and source-domain identifiers only. Held-out target data are not part of the fit API, so this is a strict Protocol-1 method.

::: neureptrace.decoding.source_vrex
    options:
      members:
        - TorchVRExClassifier
        - SourceVRExFitResult
        - fit_source_vrex_predict_proba
