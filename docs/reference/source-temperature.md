# Source temperature scaling

`neureptrace.decoding.source_temperature` implements strict source-only scalar temperature scaling for probability rows.

The protocol is **Category 1 / strict source-only**. The temperature is selected from source validation probabilities and source labels only. Held-out probability rows are transformed but are not used for fitting.

::: neureptrace.decoding.source_temperature
    options:
      members:
        - SourceTemperatureConfig
        - SourceTemperatureResult
        - fit_source_temperature_scaling
        - apply_temperature
        - negative_log_likelihood
        - source_temperature_config
