# Kernel mean matching

`neureptrace.decoding.kernel_mean_matching` implements dependency-light kernel mean matching (KMM) for unlabeled target-adaptive source weighting.

KMM estimates non-negative source-row weights so the weighted source feature distribution better matches an unlabeled target feature distribution in a kernel space. This is useful for covariate-shift correction in cross-subject decoding.

Protocol interpretation:

- uses source features `X_s`,
- optionally uses source labels `y_s` for class-balanced post-normalization,
- uses unlabeled target features `X_t`,
- does **not** accept held-out target labels `y_t`.

Therefore, KMM is a **Protocol 2 / unlabeled target-adaptive** method.

Typical use:

```python
from neureptrace.decoding.kernel_mean_matching import kernel_mean_matching_weights

result = kernel_mean_matching_weights(
    X_source,
    X_target_unlabeled,
    kernel="rbf",
    gamma="median",
    max_weight=10.0,
)

sample_weight = result.weights
```

::: neureptrace.decoding.kernel_mean_matching
    options:
      members:
        - KernelMeanMatchingConfig
        - KernelMeanMatchingResult
        - kernel_mean_matching_weights
        - kernel_mean_matching_weights_from_config
        - kmm_config
        - normalize_kmm_kernel
        - normalize_kmm_epsilon
        - resolve_kmm_gamma
