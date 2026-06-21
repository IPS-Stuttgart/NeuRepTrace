# Riemannian Procrustes Analysis

NeuRepTrace exposes full Riemannian Procrustes Analysis (RPA) for covariance-matrix transfer through `neureptrace.decoding.riemannian`.

The implementation works on trial-level SPD covariance matrices and keeps target labels out of the public prediction API. The default mode is therefore a Category 2 target-adaptive protocol: source labels train only the downstream classifier, while unlabeled target covariance matrices may estimate target geometry.

## Geometry

`riemannian_procrustes_transfer_features` applies three steps before tangent-space classification:

1. **Recenter** each source domain and the target domain to the identity by whitening their log-Euclidean covariance means.
2. **Stretch** each source domain in the SPD tangent chart so its dispersion matches the unlabeled target dispersion.
3. **Rotate** source tangent vectors only when paired covariance anchors are explicitly supplied with `rotation_mode="paired"`.

Without paired anchors, the method performs the recenter + stretch part of RPA and records `rotation_mode="none"` in the per-domain provenance.

## Protocol status

| Setting | Target features | Target labels | Protocol |
| --- | ---: | ---: | --- |
| `rotation_mode="none"` | Yes | No | Category 2 |
| `rotation_mode="paired"` with separate unlabeled shared-stimulus anchors | Yes | No | Category 2, anchor-adaptive |
| `rotation_mode="paired"` with labeled target calibration prototypes | Yes | Calibration labels used by caller | Category 3 |

The RPA functions deliberately do not accept `target_labels`. If a caller builds paired target anchors from labeled target calibration rows, that outer workflow must be reported as supervised/calibrated target alignment even though the low-level RPA transform only receives covariance matrices.

## Minimal use

```python
from neureptrace.decoding.riemannian import fit_predict_riemannian_procrustes

classifier, transfer, predictions = fit_predict_riemannian_procrustes(
    source_covariances,
    source_labels,
    target_covariances,
    source_domains=source_subject_ids,
    rotation_mode="none",
    tangent_reference_scope="source_target",
)
```

For paired rotation, provide source and target covariance anchors with matching row order:

```python
classifier, transfer, predictions = fit_predict_riemannian_procrustes(
    source_covariances,
    source_labels,
    target_covariances,
    source_domains=source_subject_ids,
    source_anchor_covariances_by_domain={"sub-01": source_anchor_covs_01},
    target_anchor_covariances=target_anchor_covs,
    rotation_mode="paired",
)
```

The returned `RiemannianProcrustesTransferResult` stores aligned source and target covariance matrices, tangent features, the target reference, target dispersion, per-source-domain stretch factors, rotations, and protocol flags.
