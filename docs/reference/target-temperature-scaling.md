# Target temperature scaling

`neureptrace.decoding.target_temperature_scaling` fits one scalar temperature on a labeled target calibration subset and applies it to separate scored probability rows.

The protocol is **Category 3 / supervised target calibration** because target calibration labels are used for fitting. Scored target labels are not accepted by the API.

Typical usage:

```python
from neureptrace.decoding.target_temperature_scaling import fit_target_temperature_scaling

result = fit_target_temperature_scaling(
    calibration_probabilities=calibration_probabilities,
    calibration_labels=calibration_labels,
    probabilities=scored_probabilities,
    classes=classes,
)

scaled_probabilities = result.probabilities
```

::: neureptrace.decoding.target_temperature_scaling
    options:
      members:
        - TargetTemperatureConfig
        - TargetTemperatureResult
        - fit_target_temperature_scaling
        - target_temperature_config
        - apply_temperature_to_probabilities
        - negative_log_likelihood
