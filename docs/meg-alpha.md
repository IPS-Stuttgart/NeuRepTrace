# MEG alpha analyses

NeuRepTrace provides reusable MEG alpha utilities under `neureptrace.meg`.
They are intended for FieldTrip-like raw/trial MATLAB structures containing
`trial`, `time`, `label`, `trialinfo`, and `grad.chanpos` fields.

## Per-trial alpha metrics

The alpha metrics workflow filters selected channels to an alpha band, extracts
the analytic signal, and writes one row per trial. Rows include alpha power,
log-alpha power, phase concentration, planar phase-fit quality, spatial phase
frequency, estimated propagation speed, and dominant phase-gradient direction on
a projected sensor plane.

```bash
neureptrace-alpha metrics \
  --mat data/Part10Data.mat \
  --root-path data,0 \
  --output outputs/part10_alpha_metrics.csv
```

For participant-file conventions such as `Part{participant}Data.mat` and
`Part{participant}CueData.mat`, use `--data-dir` and `--participant`:

```bash
neureptrace-alpha metrics \
  --data-dir data \
  --participant 10 \
  --output outputs/part10_alpha_metrics.csv
```

Defaults mirror the original PyMEGDec exploratory workflow: occipital CTF
channels matching `^M[LRZ]O`, a prestimulus window of `-0.4,-0.05` seconds, and
an `8,12` Hz alpha band.

## Sensor-level alpha movement

The movement workflow computes alpha power for selected MEG sensors and tracks
the power-weighted centroid across `data.grad.chanpos`. The resulting trajectory
is a sensor-array proxy for alpha-topography movement; it is not a source-localized
anatomical movement estimate.

```bash
neureptrace-alpha movement \
  --data-dir data \
  --participants 10 \
  --trajectory-output outputs/part10_alpha_movement.csv \
  --summary-output outputs/part10_alpha_movement_summary.csv
```

Projected centroids use the same deterministic sensor-plane projection as the
phase-gradient metrics. The projection plane is fitted from selected reference
channels and the in-plane axes are anchored to the original sensor coordinate
frame to avoid arbitrary SVD sign or rotation flips.
